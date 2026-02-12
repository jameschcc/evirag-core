# ==============================================================================
# EviRAG — Enterprise-grade High-performance Offline Knowledge Base with
# Traceable Retrieval and Generation Architecture
#         DOI: 10.3969/j.issn.1671-3370.2026.02.032 (S.Chang) https://d.wanfangdata.com.cn/periodical/zgxxj202602032
# ==============================================================================

from __future__ import annotations

import json
import re

import httpx
from pydantic import BaseModel, Field, ValidationError

from .contracts import QueryAnalysis
from .llm_client import StructuredChatClient, safe_chat_failure_reason


_REVISION = "query-planner-v3"
_CHINESE_STOP_WORDS = (
    "关于", "请问", "请介绍", "介绍", "说明", "什么", "哪些", "如何", "怎么", "是否",
    "可以", "能够", "能否", "请", "一下", "的", "了", "吗", "呢", "么", "我", "你", "我们",
    "有关", "相关", "目前", "当前", "详细", "一下子",
)
_ENGLISH_STOP_WORDS = {
    "a", "an", "and", "are", "can", "could", "do", "does", "for", "how", "is", "of", "on",
    "please", "tell", "the", "this", "to", "what", "which", "with", "about", "explain",
}
_SYNONYMS = {
    "过温": ("过热", "温度异常"),
    "入库": ("文档入库", "索引"),
    "检索": ("搜索", "召回"),
    "回答": ("问答", "生成"),
    "组件": ("模块", "服务"),
}
_TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]*|[0-9]+(?:\.[0-9]+)?|[一-鿿]+")
# 追问常见指代词；命中时把上一轮问题并入检索表达，避免只拿代词去召回。
_ANAPHORA_PATTERN = re.compile(
    r"(它|他|她|它们|这个|那个|这些|那些|该|此|其|上述|前面|刚才|上面|这种|这类|其中|这里|那里|还有|那么|这|那)"
)


class AiQueryDraft(BaseModel):
    keywords: list[str] = Field(default_factory=list, max_length=20)
    identifiers: list[str] = Field(default_factory=list, max_length=12)
    retrieval_queries: list[str] = Field(default_factory=list, max_length=3)
    # 有对话历史时，模型把追问改写成不依赖上下文的完整问题。
    resolved_question: str = Field(default="", max_length=400)
    # 刻意用 str 接住模型输出：模型常写 greeting / 闲聊 / 寒暄 这类别名，
    # 若这里用 Literal，一个别名就会让整条 draft 校验失败并连带丢掉 keywords。
    intent: str = "kb_question"


_CHITCHAT_ALIASES = {
    "chitchat", "chit_chat", "smalltalk", "small_talk", "greeting", "greet",
    "chat", "casual", "闲聊", "寒暄", "问候", "打招呼",
}
_META_ALIASES = {
    "meta", "about_self", "about_system", "self", "capability", "capabilities",
    "元问题", "自我介绍", "能力介绍", "关于系统",
}


def normalize_intent(value: object) -> str:
    """Normalize a model-produced intent; anything unrecognized means retrieval."""
    text = re.sub(r"[\s\-]+", "_", str(value or "").strip().lower())
    if text in _CHITCHAT_ALIASES:
        return "chitchat"
    if text in _META_ALIASES:
        return "meta"
    return "kb_question"


def _unique(values: list[str], limit: int = 20) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip()
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        result.append(normalized)
        if len(result) >= limit:
            break
    return result


def _last_user_question(history: list[dict] | None) -> str:
    for item in reversed(history or []):
        if str(item.get("role") or "") != "user":
            continue
        content = re.sub(r"\s+", " ", str(item.get("content") or "")).strip()
        if content:
            return content
    return ""


def _merge_followup(question: str, history: list[dict] | None) -> str:
    """Rule fallback for follow-ups: make pronoun/短追问 retrievable.

    Only applies when the question clearly depends on the previous turn
    (pronouns or a very short question); a fresh standalone question in an
    ongoing thread must not be polluted by the previous topic.
    """
    compact = re.sub(r"\s+", " ", (question or "").strip())
    previous = _last_user_question(history)
    if not compact or not previous:
        return ""
    if _ANAPHORA_PATTERN.search(compact) or len(compact) <= 6:
        merged = f"{previous} {compact}".strip()
        return merged[:400]
    return ""


def _history_block(history: list[dict] | None) -> str:
    """Compact history for the query-understanding call (not for answers)."""
    if not history:
        return ""
    lines: list[str] = []
    for item in history[-4:]:
        role = str(item.get("role") or "")
        content = re.sub(r"\s+", " ", str(item.get("content") or "")).strip()
        if role not in {"user", "assistant"} or not content:
            continue
        cap = 260 if role == "assistant" else 200
        lines.append(f"{'用户' if role == 'user' else '助手'}：{content[:cap]}")
    if not lines:
        return ""
    return "对话历史：\n" + "\n".join(lines) + "\n\n"


class QueryPlanner:
    """Use a bounded AI query-understanding call, with an inspectable rule fallback."""

    revision = _REVISION

    def __init__(
        self,
        base_url: str = "",
        model: str = "qwen3:14b",
        timeout: float = 20.0,
        provider: str = "ollama",
        api_key: str = "",
        structured_mode: str | None = None,
        tool_choice: str | None = None,
    ) -> None:
        self.client = StructuredChatClient(
            provider, base_url, api_key, model,
            structured_mode=structured_mode, tool_choice=tool_choice,
        )
        self.model = model
        self.timeout = timeout

    async def build_async(
        self,
        question: str,
        enabled: bool = True,
        supplied: QueryAnalysis | None = None,
        history: list[dict] | None = None,
    ) -> QueryAnalysis:
        original = question.strip()
        history = history or []
        if not enabled:
            return self.build(original, enabled=False, history=history)
        # A supplied analysis is a same-question cached plan from a previous
        # search call; it has no conversation rewriting, so it is only reused
        # for single-turn requests.
        if supplied and not history and supplied.original_question.strip() == original and supplied.enabled:
            return supplied
        if not self.client.available:
            return self._fallback(original, "语义分析服务未配置，已使用规则分析。", history)
        try:
            draft = await self._analyze_with_ai(original, history)
            return self._from_ai(original, draft, history)
        except (httpx.HTTPError, OSError, ValueError, ValidationError, TypeError, KeyError) as exc:
            reason = safe_chat_failure_reason(exc)
            return self._fallback(original, f"AI 语义分析不可用（{reason}），已使用规则分析。", history)

    def build(self, question: str, enabled: bool = True, history: list[dict] | None = None) -> QueryAnalysis:
        original = question.strip()
        if not enabled:
            # 追问即使关闭语义分析也必须补全指代，否则“它支持什么”只能拿代词召回。
            resolved = _merge_followup(original, history)
            return QueryAnalysis(
                enabled=False,
                original_question=original,
                resolved_question=resolved,
                retrieval_query=original,
                retrieval_queries=_unique([resolved, original], limit=3) if resolved else [original],
                analysis_method="disabled",
                revision="disabled",
            )

        keywords = self._rule_keywords(original)
        expansions = _unique([synonym for keyword in keywords for synonym in _SYNONYMS.get(keyword, ())], limit=12)
        identifiers = _unique(
            [
                term for term in re.findall(r"(?<![A-Za-z])[A-Za-z0-9][A-Za-z0-9_.:/-]*", original)
                if any(char.isdigit() for char in term) or "_" in term or "/" in term or ":" in term
            ],
            limit=12,
        )
        retrieval_terms = _unique(keywords + expansions + identifiers, limit=32)
        retrieval_query = " ".join(retrieval_terms) or original
        resolved = _merge_followup(original, history)
        retrieval_queries = [retrieval_query]
        if resolved:
            # Keep both the rewritten and the raw question so exact terms in
            # the follow-up still participate in lexical recall.
            retrieval_queries = _unique([resolved, *retrieval_queries], limit=3)
        return QueryAnalysis(
            original_question=original,
            resolved_question=resolved,
            retrieval_query=retrieval_query,
            keywords=keywords,
            expansions=expansions,
            identifiers=identifiers,
            retrieval_queries=retrieval_queries,
            analysis_method="rules",
            revision=self.revision,
        )

    async def _analyze_with_ai(self, question: str, history: list[dict] | None = None) -> AiQueryDraft:
        history_block = _history_block(history)
        messages = [
            {
                "role": "system",
                "content": (
                    "你是企业知识库的查询理解器，不负责回答问题。只返回 JSON。"
                    "从用户问题中提取检索所需的核心主题词、完整术语、设备或标准编号，"
                    "并给出最多 3 个简短检索表达。保留英文产品名、缩写、型号、日期和数字。"
                    "删除‘介绍、如何、什么、请问’等问法词。不要凭空增加产品、组织、数字或事实；"
                    "不要把自然语言问题改写成答案。"
                    "如果 user 消息给出了对话历史，最后一条是追问：结合历史把它改写成一个"
                    "脱离上下文也能独立检索的完整问题写入 resolved_question，补齐被省略的主语、"
                    "设备、标准编号和限定条件；不要回答问题，也不要引入历史与当前问题之外的事实。"
                    "没有对话历史时 resolved_question 留空字符串。"
                    "另外判断输入意图写入 intent，只能是三者之一："
                    "kb_question 是需要查知识库的事实、文档、设备、编号或条款问题；"
                    "chitchat 是纯寒暄、致谢、告别，不含任何主题；"
                    "meta 是问你自己或这个系统能做什么、怎么用。"
                    "只有 chitchat 或 meta 且完全不含任何可检索主题时才这样标；"
                    "凡是含「查一下 / 帮我 / 看看 / 有没有 / 更新没 / 是什么 / 怎么」等"
                    "请求具体信息的，即使以问候开头，也一律 kb_question。"
                    "例：「你好」→chitchat；「谢谢」→chitchat；「你能做什么」→meta；"
                    "「你好，请问 IAM 消息格式是什么」→kb_question；"
                    "「你好，帮我看看文档更新没」→kb_question。"
                ),
            },
            {"role": "user", "content": f"{history_block}用户问题：{question}"},
        ]
        response = await self.client.post(
            messages,
            AiQueryDraft.model_json_schema(),
            0.0,
            self.client.effective_max_tokens(400),
            self.timeout,
        )
        body = response.json()
        content = self.client.extract_content(body)
        return AiQueryDraft.model_validate(self._parse_json(content))

    def _from_ai(self, question: str, draft: AiQueryDraft, history: list[dict] | None = None) -> QueryAnalysis:
        keywords = _unique(draft.keywords, limit=20)
        identifiers = _unique(draft.identifiers, limit=12)
        retrieval_queries = _unique(draft.retrieval_queries, limit=3)
        expansions = _unique([synonym for keyword in keywords for synonym in _SYNONYMS.get(keyword, ())], limit=12)
        retrieval_terms = _unique(keywords + expansions + identifiers, limit=32)
        if not retrieval_terms and retrieval_queries:
            retrieval_terms = _unique(retrieval_queries[0].split(), limit=32)
        retrieval_query = " ".join(retrieval_terms) or question
        resolved = draft.resolved_question.strip()
        if not resolved:
            resolved = _merge_followup(question, history)
        queries = [retrieval_query]
        if resolved:
            queries = [resolved, *queries]
        retrieval_queries = _unique([*queries, *retrieval_queries], limit=3)
        return QueryAnalysis(
            original_question=question,
            resolved_question=resolved,
            retrieval_query=retrieval_query,
            keywords=keywords,
            expansions=expansions,
            identifiers=identifiers,
            retrieval_queries=retrieval_queries,
            intent=normalize_intent(draft.intent),
            analysis_method="ai",
            revision=self.revision,
        )

    def _fallback(self, question: str, warning: str, history: list[dict] | None = None) -> QueryAnalysis:
        return self.build(question, history=history).model_copy(
            update={"analysis_method": "fallback", "warnings": [warning]}
        )

    @staticmethod
    def _rule_keywords(question: str) -> list[str]:
        values: list[str] = [term for term in _SYNONYMS if term in question]
        for token in _TOKEN_PATTERN.findall(question):
            if re.fullmatch(r"[一-鿿]+", token):
                compact = token
                for stop_word in sorted(_CHINESE_STOP_WORDS, key=len, reverse=True):
                    compact = compact.replace(stop_word, " ")
                values.extend(part for part in compact.split() if len(part) >= 2)
            elif token.lower() not in _ENGLISH_STOP_WORDS and len(token) >= 2:
                values.append(token)
        return _unique(values)

    @staticmethod
    def _parse_json(content: str) -> dict:
        raw = content.strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE | re.DOTALL).strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            start, end = raw.find("{"), raw.rfind("}")
            if start < 0 or end <= start:
                raise
            parsed = json.loads(raw[start : end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("AI 语义分析返回的 JSON 不是对象")
        return parsed
