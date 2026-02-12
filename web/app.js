/* =============================================================================
 * EviRAG — Enterprise-grade High-performance Offline Knowledge Base with
 * Traceable Retrieval and Generation Architecture
 *         DOI: 10.3969/j.issn.1671-3370.2026.02.032 (S.Chang) https://d.wanfangdata.com.cn/periodical/zgxxj202602032
 * ========================================================================== */

const state = {
  debug: false,
  documents: [],
  codeDocuments: [],
  knowledgeBases: [],
  activeKnowledgeBaseId: null,
  documentsPage: {limit: 20, offset: 0, total: 0},
  codePage: {limit: 20, offset: 0, total: 0},
  serviceInfo: null,
  health: null,
  providerPlans: [],
  activeProviderPlanId: null,
  glossaryChoiceButton: null,
  glossaryChoiceBusy: false,
  lastProgressError: null,
  lastSearch: null,
  lastAnswer: null,
  answerPending: false,
  glossary: {limit: 20, offset: 0, total: 0},
  uploadQueue: [],
  uploadQueueRunning: false,
  uploadQueueSeq: 0,
  chat: {
    conversations: [],
    activeId: null,
    activeTitle: '',
    messages: [],
    sending: false,
    preview: '',
  },
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

// 诊断信息（status、verification、置信度、Warnings、追问改写）默认隐藏，
// 顶栏勾选 DEBUG 才显示；用 body class 控制，已渲染的消息无需重绘。
function applyDebugMode(enabled) {
  state.debug = Boolean(enabled);
  document.body.classList.toggle('debug-on', state.debug);
}

const API_BASE_KEY = 'kb-api-base';
const KB_ACTIVE_KEY = 'kb-active-id';
const KB_USER_KEY = 'kb-user-id';

function normalizeApiBase(value) {
  return String(value ?? '').trim().replace(/\/+$/, '');
}

// A "?api=" query parameter wins for one load and is persisted; otherwise the
// saved backend is used; empty means same origin (relative/nginx-proxied API).
function initialApiBase() {
  const fromQuery = new URLSearchParams(window.location.search).get('api');
  if (fromQuery !== null) {
    const normalized = normalizeApiBase(fromQuery);
    if (normalized) window.localStorage.setItem(API_BASE_KEY, normalized);
    else window.localStorage.removeItem(API_BASE_KEY);
    return normalized;
  }
  return normalizeApiBase(window.localStorage.getItem(API_BASE_KEY));
}

let apiBase = initialApiBase();
const baseUrl = () => apiBase || window.location.origin;
const apiUrl = (path) => (/^https?:\/\//i.test(path) ? path : `${apiBase}${path}`);
let textInputResolver = null;

function localUserId() {
  return window.localStorage.getItem(KB_USER_KEY) || '';
}

function kbPath(suffix) {
  return state.activeKnowledgeBaseId
    ? `/api/v2/knowledge-bases/${encodeURIComponent(state.activeKnowledgeBaseId)}${suffix}`
    : suffix;
}

function switchBackend(value) {
  const normalized = normalizeApiBase(value);
  if (normalized) window.localStorage.setItem(API_BASE_KEY, normalized);
  else window.localStorage.removeItem(API_BASE_KEY);
  window.location.reload();
}

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, (char) => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[char]));
}

function formatDate(value) {
  if (!value) return '-';
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? value : date.toLocaleString('zh-CN', {hour12: false});
}

function shortId(value) {
  if (!value) return '-';
  return `${value.slice(0, 8)}…${value.slice(-4)}`;
}

async function apiFetch(path, options = {}) {
  const headers = new Headers(options.headers || {});
  const userId = localUserId();
  if (userId) headers.set('X-KB-User-ID', userId);
  const response = await fetch(apiUrl(path), {...options, headers});
  const raw = await response.text();
  let data;
  try { data = raw ? JSON.parse(raw) : null; } catch { data = raw; }
  if (!response.ok) {
    const detail = data?.detail ? JSON.stringify(data.detail) : (raw || response.statusText);
    throw new Error(`${response.status} ${detail}`);
  }
  return data;
}

function showAlert(id, message, type = 'info') {
  const element = typeof id === 'string' ? $(`#${id}`) : id;
  if (!element) return;
  element.className = `alert ${type}`;
  element.textContent = message;
}

function hideAlert(id) {
  const element = typeof id === 'string' ? $(`#${id}`) : id;
  if (element) element.className = 'alert hidden';
}

function setBusy(button, busy, busyText = '处理中…') {
  if (!button) return;
  if (busy) {
    // Only remember the idle label once: progress updates call setBusy(true) repeatedly.
    if (!button.dataset.busy) button.dataset.originalText = button.textContent;
    button.dataset.busy = '1';
    button.textContent = busyText;
    button.disabled = true;
  } else {
    button.textContent = button.dataset.originalText || button.textContent;
    button.disabled = false;
    delete button.dataset.busy;
  }
}

function statusTag(status) {
  const label = {ready: 'ready', processing: 'processing', failed: 'failed'}[status] || status || 'unknown';
  return `<span class="status-tag ${escapeHtml(status || '')}">${escapeHtml(label)}</span>`;
}

function glossaryCountValue(count) {
  if (count === null || count === undefined || count === '') return null;
  const value = Number(count);
  return Number.isFinite(value) ? value : null;
}

async function loadKnowledgeBases() {
  const select = $('#knowledge-base-select');
  try {
    const result = await apiFetch('/api/v2/knowledge-bases');
    state.knowledgeBases = Array.isArray(result) ? result : [];
    const saved = window.localStorage.getItem(KB_ACTIVE_KEY);
    const selected = state.knowledgeBases.find((item) => item.kb_id === saved) || state.knowledgeBases[0];
    state.activeKnowledgeBaseId = selected?.kb_id || null;
    if (state.activeKnowledgeBaseId) window.localStorage.setItem(KB_ACTIVE_KEY, state.activeKnowledgeBaseId);
    if (select) {
      select.innerHTML = state.knowledgeBases.length
        ? state.knowledgeBases.map((item) => `<option value="${escapeHtml(item.kb_id)}">${escapeHtml(item.name)} · ${escapeHtml(item.role)}</option>`).join('')
        : '<option value="">没有可访问的知识库</option>';
      select.value = state.activeKnowledgeBaseId || '';
      select.disabled = !state.activeKnowledgeBaseId;
    }
  } catch (error) {
    state.knowledgeBases = [];
    state.activeKnowledgeBaseId = null;
    if (select) {
      select.innerHTML = '<option value="">知识库读取失败</option>';
      select.disabled = true;
    }
    showAlert('global-alert', `知识库读取失败：${error.message}`, 'error');
  }
}

async function switchKnowledgeBase(kbId) {
  if (!state.knowledgeBases.some((item) => item.kb_id === kbId)) return;
  state.activeKnowledgeBaseId = kbId;
  window.localStorage.setItem(KB_ACTIVE_KEY, kbId);
  state.documentsPage.offset = 0;
  state.glossary.offset = 0;
  state.lastSearch = null;
  state.lastAnswer = null;
  resetChatForKnowledgeBase();
  renderEvidence({evidence: []});
  renderAnswer(null);
  await Promise.all([loadDocuments(), loadGlossary()]);
}

function resetChatForKnowledgeBase() {
  state.chat.conversations = [];
  state.chat.activeId = null;
  state.chat.activeTitle = '';
  state.chat.messages = [];
  state.chat.sending = false;
  state.chat.preview = '';
  renderChatThread();
  renderConversations();
  if (state.activeKnowledgeBaseId) void loadConversations();
}

function requestTextInput({title, description, label, placeholder, submitLabel}) {
  if (textInputResolver) textInputResolver(null);
  return new Promise((resolve) => {
    textInputResolver = resolve;
    $('#text-input-title').textContent = title;
    $('#text-input-description').textContent = description;
    $('#text-input-label').textContent = label;
    $('#text-input-value').value = '';
    $('#text-input-value').placeholder = placeholder || '';
    $('#text-input-submit').textContent = submitLabel || '确定';
    $('#text-input-modal').classList.remove('hidden');
    document.body.classList.add('reader-open');
    window.setTimeout(() => $('#text-input-value').focus(), 0);
  });
}

function closeTextInput(value = null) {
  $('#text-input-modal').classList.add('hidden');
  document.body.classList.remove('reader-open');
  const resolve = textInputResolver;
  textInputResolver = null;
  if (resolve) resolve(value);
}

async function createKnowledgeBase() {
  const name = await requestTextInput({
    title: '新建知识库',
    description: '知识库之间按 RBAC 权限和 kb_id 逻辑隔离，不需要切换端口。',
    label: '知识库名称',
    placeholder: '例如：项目研发资料',
    submitLabel: '创建并切换',
  });
  if (!name || !name.trim()) return;
  try {
    const result = await apiFetch('/api/v2/knowledge-bases', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({name: name.trim()}),
    });
    await loadKnowledgeBases();
    await switchKnowledgeBase(result.kb_id);
    showAlert('global-alert', `已创建知识库“${result.name}”。`, 'success');
  } catch (error) {
    showAlert('global-alert', `创建知识库失败：${error.message}`, 'error');
  }
}

function glossaryHasEntries(glossaryState, count) {
  const knownCount = glossaryCountValue(count);
  return knownCount === null ? Number(glossaryState) === 1 : knownCount > 0;
}

function glossaryLabel(glossaryState, count, candidateCount) {
  const knownCount = glossaryCountValue(count);
  if (glossaryHasEntries(glossaryState, count)) {
    return knownCount === null ? '已收录' : `已收录 ${knownCount} 条`;
  }
  const knownCandidateCount = glossaryCountValue(candidateCount);
  if (knownCandidateCount > 0) return `候选待提取 ${knownCandidateCount} 段`;
  if (knownCount === 0 || Number(glossaryState) === -1) return '无术语库';
  return '提取术语库';
}

function glossaryClass(glossaryState, count) {
  return glossaryHasEntries(glossaryState, count) ? 'button xs glossary-done glossary-build' : 'button xs secondary glossary-build';
}

function glossaryTitle(glossaryState, count, candidateCount) {
  if (glossaryHasEntries(glossaryState, count)) return '已收录，点击选择 AI 辅助补充或按规则重新检测';
  if (Number(candidateCount) > 0) return '已发现候选段落，点击选择 AI 辅助提取或按规则重新检测';
  if (glossaryCountValue(count) === 0 || Number(glossaryState) === -1) return '点击后重新检测术语/缩写/定义段';
  return '空闲时用 AI 分析术语段并收录';
}

function glossaryButton(doc) {
  const glossaryState = Number(doc.glossary_state ?? 0);
  const glossaryCount = glossaryCountValue(doc.glossary_count);
  const candidateCount = glossaryCountValue(doc.glossary_candidate_count);
  return `<button type="button" class="${glossaryClass(glossaryState, glossaryCount)}" data-document-id="${escapeHtml(doc.document_id)}" data-document-title="${escapeHtml(doc.title || '')}" data-glossary-state="${glossaryState}" data-glossary-count="${glossaryCount ?? ''}" data-glossary-candidates="${candidateCount ?? ''}" title="${escapeHtml(glossaryTitle(glossaryState, glossaryCount, candidateCount))}">${escapeHtml(glossaryLabel(glossaryState, glossaryCount, candidateCount))}</button>`;
}

function openGlossaryChoice(button) {
  state.glossaryChoiceButton = button;
  state.glossaryChoiceBusy = false;
  const candidateCount = Number(button.dataset.glossaryCandidates || 0);
  const glossaryCount = Number(button.dataset.glossaryCount || 0);
  const title = button.dataset.documentTitle || '当前文档';
  const description = candidateCount > 0
    ? `“${title}”已发现 ${candidateCount} 个候选段落，但当前收录 ${glossaryCount} 条术语。请选择提取方式。`
    : glossaryCount > 0
      ? `“${title}”当前已收录 ${glossaryCount} 条术语。你可以使用 AI 辅助补充，或按规则重新检测。`
      : `之前未检测到术语。选择“使用 AI 辅助”会规划检索词并补充上下文；也可以按规则重新检测。`;
  $('#glossary-choice-description').textContent = description;
  $('#glossary-choice-status').textContent = '';
  $('#glossary-choice-modal').classList.remove('hidden');
  document.body.classList.add('reader-open');
  $('#glossary-choice-modal').querySelector('[data-glossary-strategy="ai_assisted"]').focus();
}

function handleGlossaryBuild(button) {
  return openGlossaryChoice(button);
}

function renderDocuments() {
  const body = $('#documents-body');
  const docs = state.documents;
  const {limit, offset, total} = state.documentsPage;
  $('#document-count').textContent = `${total} 个文档`;
  const page = Math.floor(offset / limit) + 1;
  const pages = Math.max(1, Math.ceil(total / limit));
  $('#documents-page-label').textContent = `第 ${page} / ${pages} 页`;
  $('#documents-prev').disabled = offset <= 0;
  $('#documents-next').disabled = offset + limit >= total;
  if (!docs.length) {
    body.innerHTML = '<tr><td colspan="8" class="empty-cell">暂无文档，请先上传支持的文档格式。</td></tr>';
    return;
  }
  body.innerHTML = docs.map((doc) => `<tr>
    <td><strong>${escapeHtml(doc.title || 'untitled')}</strong><small>${escapeHtml(doc.status === 'ready' ? '可检索' : '等待处理')}</small></td>
    <td><span class="format-tag">${(doc.format || "text").toUpperCase()}</span>${doc.source_kind === 'code' ? '<span class="source-tag">代码</span>' : ''}</td>
    <td>${statusTag(doc.status)}<small class="mono">task ${shortId(doc.task_id)}</small></td>
    <td><strong>${escapeHtml(doc.chunk_count ?? 0)}</strong></td>
    <td class="mono" title="${escapeHtml(doc.document_id)}">${escapeHtml(shortId(doc.document_id))}</td>
    <td class="mono" title="${escapeHtml(doc.version_id)}">${escapeHtml(shortId(doc.version_id))}</td>
    <td>${escapeHtml(formatDate(doc.created_at))}</td>
    <td><span class="error-cell">${escapeHtml(doc.error || "")}</span><button class="button secondary xs preview-document" data-document-id="${escapeHtml(doc.document_id)}" data-title="${escapeHtml(doc.title || '')}" data-format="${escapeHtml(doc.format || 'text')}">预览</button>${glossaryButton(doc)}${doc.glossary_count > 0 ? `<button class="button secondary xs view-glossary" data-document-id="${escapeHtml(doc.document_id)}" data-title="${escapeHtml(doc.title || '')}">查看术语</button>` : ''}<button class="button secondary xs use-document" data-document-id="${escapeHtml(doc.document_id)}">去检索</button></td>
  </tr>`).join('');
  body.querySelectorAll('.use-document').forEach((button) => button.addEventListener('click', () => {
    switchPage('search');
    $('#question-input').focus();
  }));
  body.querySelectorAll('.glossary-build').forEach((button) => button.addEventListener('click', () => handleGlossaryBuild(button)));
  body.querySelectorAll('.view-glossary').forEach((button) => button.addEventListener('click', () => openDocumentGlossary(button.dataset.documentId, button.dataset.title)));
  body.querySelectorAll('.preview-document').forEach((button) => button.addEventListener('click', () => openDocumentPreview(button.dataset.documentId, button.dataset.title, button.dataset.format)));
}

async function loadDocuments() {
  const {limit, offset} = state.documentsPage;
  try {
    const result = await apiFetch(kbPath(`/documents?limit=${limit}&offset=${offset}&source_kind=document`));
    state.documents = result.documents || [];
    state.documentsPage.total = result.total || 0;
    renderDocuments();
    hideAlert('global-alert');
  } catch (error) {
    $('#documents-body').innerHTML = '<tr><td colspan="8" class="empty-cell">文档列表读取失败。</td></tr>';
    showAlert('global-alert', `无法读取文档列表：${error.message}`, 'error');
  }
}

function setGlossaryChoiceBusy(busy, message = '') {
  state.glossaryChoiceBusy = busy;
  $('#glossary-choice-status').textContent = message;
  $$('#glossary-choice-modal button').forEach((item) => { item.disabled = busy; });
}

function closeGlossaryChoice() {
  if (state.glossaryChoiceBusy) return;
  $('#glossary-choice-modal').classList.add('hidden');
  document.body.classList.remove('reader-open');
  state.glossaryChoiceButton = null;
}

function glossaryAuditSummary(result) {
  const audit = result.extraction_audit || [];
  const counts = audit.reduce((total, item) => {
    total[item.status] = (total[item.status] || 0) + 1;
    if (item.error_code) total.errors += 1;
    return total;
  }, {});
  const rounds = (result.rounds || []).length;
  const terms = (result.search_terms || []).slice(0, 12).join('、');
  const parts = [
    `策略：${result.strategy === 'ai_assisted' ? 'AI 辅助' : '规则重检'}`,
    `候选段：${result.candidate_count || 0}`,
    `规则候选：${result.rule_candidate_count ?? result.candidate_count ?? 0}`,
    `补充轮次：${rounds}`,
    `抽取调用：${audit.length}（成功 ${counts.completed || 0}、空结果 ${counts.empty || 0}、异常 ${counts.errors || 0}）`,
  ];
  if (terms) parts.push(`检索词：${terms}`);
  if (result.warnings?.length) parts.push(`提示：${result.warnings.join('；')}`);
  return parts.join('；');
}

async function buildGlossary(button, strategy = 'rules') {
  const documentId = button.dataset.documentId;
  const progressId = newProgressId();
  const strategyLabel = strategy === 'ai_assisted' ? 'AI 辅助处理中…' : '规则检测中…';
  setBusy(button, true, strategyLabel);
  setGlossaryChoiceBusy(true, strategyLabel);
  let stopProgress = false;
  let progressTimer = null;
  const pollProgress = async () => {
    try {
      const progress = await apiFetch(kbPath(`/progress/${encodeURIComponent(progressId)}`));
      if (progress.message) $('#glossary-choice-status').textContent = progress.message;
    } catch { /* the extraction request remains authoritative */ }
    if (!stopProgress) progressTimer = window.setTimeout(pollProgress, 700);
  };
  pollProgress();
  try {
    const result = await apiFetch(kbPath(`/documents/${encodeURIComponent(documentId)}/glossary?strategy=${strategy}&progress_id=${encodeURIComponent(progressId)}`), {method: 'POST'});
    const count = result.entries ? result.entries.length : 0;
    const candidateCount = Number(result.candidate_count || 0);
    const auditSummary = glossaryAuditSummary(result);
    if (count > 0) {
      showAlert('global-alert', `术语库处理完成，本次新增 ${count} 条，当前共 ${result.glossary_count || count} 条。${strategy === 'ai_assisted' ? ` ${auditSummary}` : ''}`, 'success');
    } else if (candidateCount > 0) {
      showAlert('global-alert', `发现 ${candidateCount} 个候选段，但本次没有新增有效词条；已有术语不会被清除。${auditSummary}`, 'info');
    } else if (result.glossary_count > 0) {
      showAlert('global-alert', `重新检测未找到候选段，已保留现有 ${result.glossary_count} 条术语。${auditSummary}`, 'info');
    } else {
      showAlert('global-alert', `重新检测后仍未找到术语、缩写或定义候选段。${auditSummary}`, 'info');
    }
    setGlossaryChoiceBusy(false, '流程完成。');
    closeGlossaryChoice();
    await loadDocuments();
  } catch (error) {
    showAlert('global-alert', `术语库提取失败：${error.message}`, 'error');
    setBusy(button, false);
    setGlossaryChoiceBusy(false, `处理失败：${error.message}`);
  } finally {
    stopProgress = true;
    if (progressTimer) window.clearTimeout(progressTimer);
  }
}

function glossaryDeleteButton(entry) {
  return `<button class="button secondary xs remove-glossary-entry" type="button" data-document-id="${escapeHtml(entry.document_id || '')}" data-term="${escapeHtml(entry.term || '')}" data-expansion="${escapeHtml(entry.expansion || '')}" title="从术语库删除这一条对照，不影响文档、切片与历史答案">删除</button>`;
}

function glossaryRows(entries, showDocument = false) {
  return entries.map((entry) => `<tr>
    <td><strong>${escapeHtml(entry.term || '-')}</strong></td>
    <td>${escapeHtml(entry.expansion || '-')}</td>
    ${showDocument ? `<td>${escapeHtml(entry.title || '-')}<small class="mono">${escapeHtml(shortId(entry.document_id))}</small></td>` : ''}
    <td>${escapeHtml(entry.origin || '-')}</td>
    <td class="mono">${escapeHtml(shortId(entry.chunk_id))}</td>
    <td>${glossaryDeleteButton(entry)}</td>
  </tr>`).join('');
}

async function removeGlossaryEntry(button) {
  const {documentId, term, expansion} = button.dataset;
  if (!documentId || !term) return;
  if (!window.confirm(`确定从术语库删除“${term}”吗？\n只删除这一条术语对照，不影响文档、切片与历史答案。`)) return;
  setBusy(button, true, '删除中…');
  try {
    await apiFetch(kbPath(`/documents/${encodeURIComponent(documentId)}/glossary?term=${encodeURIComponent(term)}&expansion=${encodeURIComponent(expansion || '')}`), {method: 'DELETE'});
    showAlert('global-alert', `已从术语库删除“${term}”。`, 'success');
    const modalOpen = !$('#glossary-modal').classList.contains('hidden');
    const modalTitle = $('#glossary-title').textContent;
    await loadGlossary();
    await loadDocuments();
    if (modalOpen) await openDocumentGlossary(documentId, modalTitle);
  } catch (error) {
    showAlert('global-alert', `删除术语失败：${error.message}`, 'error');
    setBusy(button, false);
  }
}

async function loadGlossary() {
  const {limit, offset} = state.glossary;
  try {
    const result = await apiFetch(kbPath(`/glossary?limit=${limit}&offset=${offset}`));
    state.glossary.total = result.total || 0;
    const entries = result.entries || [];
    if (!entries.length && offset > 0 && state.glossary.total > 0) {
      // 删掉当前页最后一条后回退一页，而不是停在空页。
      state.glossary.offset = Math.max(0, offset - limit);
      return loadGlossary();
    }
    $('#glossary-body').innerHTML = entries.length
      ? glossaryRows(entries, true)
      : '<tr><td colspan="6" class="empty-cell">术语库为空，请先在文档列表点击“提取术语库”。</td></tr>';
    $('#glossary-total').textContent = `${state.glossary.total} 条`;
    const page = Math.floor(offset / limit) + 1;
    const pages = Math.max(1, Math.ceil(state.glossary.total / limit));
    $('#glossary-page-label').textContent = `第 ${page} / ${pages} 页`;
    $('#glossary-prev').disabled = offset <= 0;
    $('#glossary-next').disabled = offset + limit >= state.glossary.total;
    hideAlert('glossary-alert');
  } catch (error) {
    $('#glossary-body').innerHTML = '<tr><td colspan="6" class="empty-cell">术语库读取失败。</td></tr>';
    showAlert('glossary-alert', `无法读取术语库：${error.message}`, 'error');
  }
}

function setGlossaryOpen(open) {
  $('#glossary-modal').classList.toggle('hidden', !open);
  document.body.classList.toggle('reader-open', open);
  if (!open) $('#glossary-entries').innerHTML = '';
}

async function openDocumentGlossary(documentId, title) {
  $('#glossary-title').textContent = title || '文档术语库';
  $('#glossary-subtitle').textContent = '';
  $('#glossary-status').textContent = '正在读取术语库…';
  $('#glossary-entries').innerHTML = '';
  setGlossaryOpen(true);
  try {
    const result = await apiFetch(kbPath(`/documents/${encodeURIComponent(documentId)}/glossary`));
    const entries = result.entries || [];
    $('#glossary-subtitle').textContent = `${result.title || title || ''} · ${entries.length} 条`;
    $('#glossary-entries').innerHTML = entries.length
      ? `<table class="glossary-table"><thead><tr><th>术语</th><th>全称 / 展开</th><th>来源</th><th>Chunk</th><th>操作</th></tr></thead><tbody>${glossaryRows(entries)}</tbody></table>`
      : '<p class="empty-state">该文档暂无术语条目。</p>';
    $('#glossary-status').textContent = entries.length ? '以下术语用于查询期缩写展开。' : '未收录术语。';
  } catch (error) {
    $('#glossary-status').textContent = `术语库读取失败：${error.message}`;
  }
}

const MAX_DOCUMENT_UPLOAD_BYTES = 50 * 1024 * 1024;
const DOCUMENT_EXTENSIONS = new Set(['.pdf', '.docx', '.pptx', '.txt', '.md', '.csv']);
const DOCUMENT_UPLOAD_REJECT_REASONS = {code: '代码文件，请到「代码库」页入库', too_large: '超过 50 MiB', unsupported: '格式不支持', empty: '空文件'};
const DOCUMENT_UPLOAD_STATUS_TEXT = {pending: '等待中', uploading: '上传中…', done: '已接收，后台解析中', failed: '失败'};

function documentExtension(name) {
  const lowered = String(name || '').toLowerCase();
  const dot = lowered.lastIndexOf('.');
  return dot >= 0 ? lowered.slice(dot) : '';
}

// Pure: null means the file may enter the queue, otherwise the reject reason.
function documentUploadRejectReason(file) {
  if (!file || !file.name) return 'empty';
  if (isCodePath(file.name)) return 'code';
  if (Number(file.size) > MAX_DOCUMENT_UPLOAD_BYTES) return 'too_large';
  if (!DOCUMENT_EXTENSIONS.has(documentExtension(file.name))) return 'unsupported';
  if (!Number(file.size)) return 'empty';
  return null;
}

function formatFileSize(bytes) {
  const value = Math.max(0, Number(bytes) || 0);
  if (value >= 1024 * 1024) return `${(value / 1024 / 1024).toFixed(2)} MiB`;
  if (value >= 1024) return `${Math.round(value / 1024)} KiB`;
  return `${value} B`;
}

function uploadQueueCounts(items) {
  const counts = {pending: 0, uploading: 0, done: 0, failed: 0};
  (items || []).forEach((item) => { counts[item.status] = (counts[item.status] || 0) + 1; });
  return counts;
}

function uploadQueueSummaryText(items) {
  const counts = uploadQueueCounts(items);
  const parts = [];
  if (counts.uploading) parts.push(`处理中 ${counts.uploading}`);
  if (counts.pending) parts.push(`等待 ${counts.pending}`);
  if (counts.done) parts.push(`完成 ${counts.done}`);
  if (counts.failed) parts.push(`失败 ${counts.failed}`);
  return parts.length ? parts.join(' · ') : '队列为空';
}

function renderUploadQueue() {
  const box = $('#upload-queue');
  if (!box) return;
  const items = state.uploadQueue;
  box.classList.toggle('hidden', !items.length);
  if (!items.length) { box.innerHTML = ''; return; }
  const counts = uploadQueueCounts(items);
  const rows = items.map((item) => {
    const detail = item.status === 'failed'
      ? `失败：${item.message || '未知错误'}`
      : (item.message || DOCUMENT_UPLOAD_STATUS_TEXT[item.status] || item.status);
    const action = item.status === 'failed' && item.file
      ? `<button class="button secondary xs" type="button" data-queue-retry="${escapeHtml(item.id)}">重试</button>`
      : item.status === 'pending'
        ? `<button class="button secondary xs" type="button" data-queue-remove="${escapeHtml(item.id)}">移除</button>`
        : '';
    return `<li class="upload-queue-item ${escapeHtml(item.status)}"><span class="upload-queue-name" title="${escapeHtml(item.name)}">${escapeHtml(item.name)}</span><span class="upload-queue-size">${escapeHtml(formatFileSize(item.size))}</span><span class="upload-queue-status" title="${escapeHtml(detail)}">${escapeHtml(detail)}</span>${action}</li>`;
  }).join('');
  box.innerHTML = `<div class="upload-queue-head"><strong>上传队列</strong><span>${escapeHtml(uploadQueueSummaryText(items))}</span>${counts.done || counts.failed ? '<button class="button secondary xs" type="button" data-queue-clear-done>清空已结束</button>' : ''}</div><ol class="upload-queue-list">${rows}</ol>`;
}

function uploadQueuePendingItem() {
  return state.uploadQueue.find((item) => item.status === 'pending') || null;
}

function handleUploadQueueAction(event) {
  if (event.target.closest('[data-queue-clear-done]')) {
    state.uploadQueue = state.uploadQueue.filter((item) => item.status === 'pending' || item.status === 'uploading');
    renderUploadQueue();
    return;
  }
  const remove = event.target.closest('[data-queue-remove]');
  if (remove) {
    state.uploadQueue = state.uploadQueue.filter((item) => item.id !== remove.dataset.queueRemove);
    renderUploadQueue();
    return;
  }
  const retry = event.target.closest('[data-queue-retry]');
  if (!retry) return;
  const item = state.uploadQueue.find((entry) => entry.id === retry.dataset.queueRetry);
  if (!item || item.status !== 'failed' || !item.file) return;
  item.status = 'pending';
  item.message = '';
  renderUploadQueue();
  startUploadQueue();
}

function enqueueDocumentFiles(fileList) {
  const files = [...(fileList || [])];
  if (!files.length) return;
  const rejected = [];
  let added = 0;
  files.forEach((file) => {
    const reason = documentUploadRejectReason(file);
    if (reason) { rejected.push(reason); return; }
    state.uploadQueue.push({id: `upload-${++state.uploadQueueSeq}`, file, name: file.name, size: Number(file.size) || 0, status: 'pending', message: ''});
    added += 1;
  });
  renderUploadQueue();
  if (rejected.length) {
    const counts = rejected.reduce((total, reason) => { total[reason] = (total[reason] || 0) + 1; return total; }, {});
    const detail = Object.entries(counts).map(([reason, count]) => `${DOCUMENT_UPLOAD_REJECT_REASONS[reason] || reason} ${count} 个`).join('、');
    showAlert('global-alert', `已跳过 ${rejected.length} 个文件（${detail}）${added ? `，其余 ${added} 个已加入上传队列。` : '。'}`, 'info');
  }
  startUploadQueue();
}

// Upload strictly one file at a time so a multi-file drop becomes a queue instead of a burst of requests.
async function startUploadQueue() {
  if (state.uploadQueueRunning || !uploadQueuePendingItem()) return;
  state.uploadQueueRunning = true;
  try {
    for (let item = uploadQueuePendingItem(); item; item = uploadQueuePendingItem()) {
      item.status = 'uploading';
      item.message = '';
      renderUploadQueue();
      try {
        const form = new FormData();
        form.append('file', item.file);
        const result = await apiFetch(kbPath('/documents'), {method: 'POST', body: form});
        item.status = 'done';
        item.documentId = result?.document_id || '';
        item.file = null; // done: release the blob so a long queue does not hold every file in memory
      } catch (error) {
        item.status = 'failed';
        item.message = error.message;
      }
      renderUploadQueue();
      state.documentsPage.offset = 0;
      await loadDocuments();
    }
  } finally {
    state.uploadQueueRunning = false;
  }
  const counts = uploadQueueCounts(state.uploadQueue);
  if (counts.done || counts.failed) {
    showAlert('global-alert', `上传队列处理完成：成功 ${counts.done} 个${counts.failed ? `，失败 ${counts.failed} 个（可在队列中重试）` : ''}。文档在后台解析与切片，状态变为 ready 后即可检索。`, counts.failed ? 'error' : 'success');
  }
}

const CODE_EXTENSIONS = new Set(['.py','.pyi','.js','.mjs','.cjs','.jsx','.ts','.tsx','.go','.java','.kt','.kts','.c','.h','.cc','.cpp','.cxx','.hpp','.hh','.cs','.rs','.rb','.php','.phtml','.sh','.bash','.zsh','.ps1','.bat','.cmd','.sql','.swift','.m','.mm','.scala','.lua','.pl','.pm','.r','.vue','.svelte','.dart','.ex','.exs','.erl','.hrl','.clj','.cljs','.groovy','.gradle','.json','.yaml','.yml','.toml','.ini','.cfg','.conf','.xml','.html','.htm','.css','.scss','.less','.mod','.sum']);
const CODE_BASENAMES = new Set(['dockerfile','makefile','jenkinsfile','vagrantfile','gemfile','rakefile','procfile','.gitignore','.dockerignore','.editorconfig']);
const CODE_IGNORED_DIRS = new Set(['.git','.svn','.hg','node_modules','__pycache__','.venv','venv','env','dist','build','target','vendor','.idea','.vscode','.next','.nuxt','coverage','.pytest_cache','.mypy_cache','.ruff_cache','__MACOSX']);
const CODE_SKIP_REASONS = {not_code:'非代码文件', too_large:'超过单文件上限', binary:'二进制文件', empty:'空文件', unsafe_path:'路径不安全', ignored_dir:'忽略目录', unreadable:'无法读取'};

function codePathParts(path) {
  return String(path || '').replace(/\\/g, '/').split('/').filter((part) => part && part !== '.');
}

function isCodePath(path) {
  const parts = codePathParts(path);
  if (!parts.length) return false;
  const name = parts[parts.length - 1].toLowerCase();
  if (CODE_BASENAMES.has(name)) return true;
  const dot = name.lastIndexOf('.');
  return dot >= 0 && CODE_EXTENSIONS.has(name.slice(dot));
}

function filterCodeFiles(files) {
  return [...files].filter((file) => {
    const path = file.webkitRelativePath || file.name;
    const parts = codePathParts(path);
    return isCodePath(path) && !parts.slice(0, -1).some((part) => CODE_IGNORED_DIRS.has(part));
  });
}

function codeSkipSummary(skipped) {
  const counts = skipped.reduce((total, item) => {
    total[item.reason] = (total[item.reason] || 0) + 1;
    return total;
  }, {});
  return Object.entries(counts).map(([reason, count]) => `${CODE_SKIP_REASONS[reason] || reason} ${count}`).join('、');
}

function renderCodeDocuments() {
  const body = $('#code-body');
  const docs = state.codeDocuments;
  const {limit, offset, total} = state.codePage;
  $('#code-count').textContent = `${total} 个文件`;
  const page = Math.floor(offset / limit) + 1;
  const pages = Math.max(1, Math.ceil(total / limit));
  $('#code-page-label').textContent = `第 ${page} / ${pages} 页`;
  $('#code-prev').disabled = offset <= 0;
  $('#code-next').disabled = offset + limit >= total;
  if (!docs.length) {
    body.innerHTML = '<tr><td colspan="8" class="empty-cell">代码库为空。选择代码文件夹或上传 ZIP 开始入库。</td></tr>';
    return;
  }
  body.innerHTML = docs.map((doc) => `<tr>
    <td><strong>${escapeHtml(doc.title || 'untitled')}</strong><small>${escapeHtml(doc.status === 'ready' ? '可检索' : '等待处理')}</small></td>
    <td><span class="format-tag">${(doc.format || "text").toUpperCase()}</span><span class="source-tag">代码</span></td>
    <td>${statusTag(doc.status)}<small class="mono">task ${shortId(doc.task_id)}</small></td>
    <td><strong>${escapeHtml(doc.chunk_count ?? 0)}</strong></td>
    <td class="mono" title="${escapeHtml(doc.document_id)}">${escapeHtml(shortId(doc.document_id))}</td>
    <td class="mono" title="${escapeHtml(doc.version_id)}">${escapeHtml(shortId(doc.version_id))}</td>
    <td>${escapeHtml(formatDate(doc.created_at))}</td>
    <td><span class="error-cell">${escapeHtml(doc.error || "")}</span><button class="button secondary xs preview-document" data-document-id="${escapeHtml(doc.document_id)}" data-title="${escapeHtml(doc.title || '')}" data-format="${escapeHtml(doc.format || 'text')}">预览</button><button class="button secondary xs use-document" data-document-id="${escapeHtml(doc.document_id)}">去检索</button></td>
  </tr>`).join('');
  body.querySelectorAll('.use-document').forEach((button) => button.addEventListener('click', () => {
    switchPage('search');
    $('#question-input').focus();
  }));
  body.querySelectorAll('.preview-document').forEach((button) => button.addEventListener('click', () => openDocumentPreview(button.dataset.documentId, button.dataset.title, button.dataset.format)));
}

async function loadCodeDocuments() {
  const {limit, offset} = state.codePage;
  try {
    const result = await apiFetch(kbPath(`/documents?limit=${limit}&offset=${offset}&source_kind=code`));
    state.codeDocuments = result.documents || [];
    state.codePage.total = result.total || 0;
    renderCodeDocuments();
    hideAlert('code-alert');
  } catch (error) {
    $('#code-body').innerHTML = '<tr><td colspan="8" class="empty-cell">代码列表读取失败。</td></tr>';
    showAlert('code-alert', `无法读取代码列表：${error.message}`, 'error');
  }
}

async function uploadCodeFolder(fileList) {
  const files = filterCodeFiles(fileList);
  const button = $('#choose-code-folder');
  const selected = $('#code-selected');
  if (!files.length) {
    selected.className = 'selected-file hidden';
    showAlert('code-alert', '没有识别到可入库的代码文件（已跳过忽略目录与非代码文件）。', 'info');
    return;
  }
  selected.className = 'selected-file';
  selected.textContent = `${files.length} 个代码文件待上传`;
  const batchSize = 50;
  const accepted = [];
  const skipped = [];
  setBusy(button, true, `上传 0/${files.length}`);
  try {
    for (let start = 0; start < files.length; start += batchSize) {
      const batch = files.slice(start, start + batchSize);
      const form = new FormData();
      batch.forEach((file) => {
        form.append('files', file, file.name);
        form.append('paths', file.webkitRelativePath || file.name);
      });
      const result = await apiFetch(kbPath('/code/uploads'), {method: 'POST', body: form});
      accepted.push(...(result.accepted || []));
      skipped.push(...(result.skipped || []));
      setBusy(button, true, `上传 ${Math.min(start + batchSize, files.length)}/${files.length}`);
    }
    const replaced = accepted.filter((item) => item.replaced).length;
    const skippedText = skipped.length ? `；跳过 ${skipped.length} 个（${codeSkipSummary(skipped)}）` : '';
    showAlert('code-alert', `已接收 ${accepted.length} 个代码文件${replaced ? `，其中 ${replaced} 个更新了原文档` : ''}${skippedText}。`, 'success');
    state.codePage.offset = 0;
    await loadCodeDocuments();
  } catch (error) {
    showAlert('code-alert', `代码入库失败：${error.message}`, 'error');
  } finally {
    setBusy(button, false);
    $('#code-folder-input').value = '';
  }
}

async function uploadCodeZip(file) {
  if (!file) return;
  if (!/\.zip$/i.test(file.name)) {
    showAlert('code-alert', '请选择 .zip 压缩包。', 'error');
    return;
  }
  const button = $('#choose-code-zip');
  const selected = $('#code-selected');
  selected.className = 'selected-file';
  selected.textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MiB`;
  setBusy(button, true, '解压并入库…');
  try {
    const form = new FormData();
    form.append('file', file);
    const result = await apiFetch(kbPath('/code/zip'), {method: 'POST', body: form});
    const accepted = result.accepted || [];
    const skipped = result.skipped || [];
    const replaced = accepted.filter((item) => item.replaced).length;
    const skippedText = skipped.length ? `；跳过 ${skipped.length} 个（${codeSkipSummary(skipped)}）` : '';
    showAlert('code-alert', `ZIP 解压完成，已接收 ${accepted.length} 个代码文件${replaced ? `，其中 ${replaced} 个更新了原文档` : ''}${skippedText}。`, 'success');
    state.codePage.offset = 0;
    await loadCodeDocuments();
  } catch (error) {
    showAlert('code-alert', `ZIP 入库失败：${error.message}`, 'error');
  } finally {
    setBusy(button, false);
    $('#code-zip-input').value = '';
  }
}

function highlightEvidence(text, question) {
  const value = String(text || '');
  const stopWords = new Set(['如何', '怎么', '什么', '哪些', '是否', '可以', '请问', '一下', '我们', '你们', '这个', '那个', '的', '了', '吗']);
  const segments = typeof Intl.Segmenter === 'function'
    ? [...new Intl.Segmenter('zh', {granularity: 'word'}).segment(question || '')].filter((part) => part.isWordLike).map((part) => part.segment)
    : (question || '').match(/[\p{L}\p{N}]+/gu) || [];
  const terms = [...new Set(segments.map((term) => term.toLowerCase()))].filter((term) => term.length > 1 && !stopWords.has(term));
  const lower = value.toLowerCase();
  const ranges = [];
  for (const term of terms) {
    let start = lower.indexOf(term);
    while (start !== -1) {
      ranges.push([start, start + term.length]);
      start = lower.indexOf(term, start + term.length);
    }
  }
  ranges.sort((a, b) => a[0] - b[0] || b[1] - a[1]);
  const merged = [];
  for (const range of ranges) {
    const previous = merged[merged.length - 1];
    if (previous && range[0] <= previous[1]) previous[1] = Math.max(previous[1], range[1]);
    else merged.push([...range]);
  }
  let offset = 0;
  let html = '';
  for (const [start, end] of merged) {
    html += escapeHtml(value.slice(offset, start)) + `<mark class="evidence-match">${escapeHtml(value.slice(start, end))}</mark>`;
    offset = end;
  }
  return html + escapeHtml(value.slice(offset));
}

function renderEvidencePending() {
  state.lastSearch = null;
  state.lastAnswer = null;
  state.answerPending = false;
  const list = $('#evidence-list');
  if (list) list.innerHTML = '<div class="card empty-state">正在检索证据…</div>';
  const revision = $('#retrieval-revision');
  if (revision) revision.textContent = '';
  const summary = $('#retrieval-summary');
  if (summary) summary.textContent = '正在检索证据，请稍候…';
  const analysis = $('#query-analysis-summary');
  if (analysis) analysis.textContent = '';
}

function renderEvidence(result) {
  state.lastSearch = result;
  const list = $('#evidence-list');
  const expanded = new Set([...list.querySelectorAll('details[open]')].map((item) => item.id));
  const evidence = result?.evidence || [];
  $('#retrieval-revision').textContent = result?.pipeline_revision ? `流水线 ${result.pipeline_revision}` : '';
  const trace = result?.iterative_trace || [];
  const addedCount = trace.reduce((sum, item) => sum + (item.added || 0), 0);
  const iterativeText = trace.length ? ` · 迭代 ${trace.length} 轮 +${addedCount} 段` : '';
  $('#retrieval-summary').textContent = `${evidence.length} 条证据 · ${result?.degraded ? '已降级' : '正常'}${iterativeText}${result?.warnings?.length ? ` · ${result.warnings.length} 条警告` : ''}`;
  const analysis = result?.query_analysis;
  const rawErrorText = !state.answerPending ? renderRawErrorText(result?.raw_error_text) : '';
  const glossary = (analysis?.glossary) || [];
  const glossaryText = glossary.length ? ` · 术语展开：${glossary.map((item) => `${item.term}→${item.expansion}`).join('、')}` : '';
  $('#query-analysis-summary').textContent = analysis
    ? (analysis.enabled
      ? `语义分析（${analysis.analysis_method === 'ai' ? 'AI' : '规则兜底'}）：${(analysis.keywords || []).join('、') || '未提取到主题词'}${(analysis.expansions || []).length ? ` · 受控扩展：${analysis.expansions.join('、')}` : ''}${glossaryText} · 实际关键词检索：${analysis.retrieval_query}`
      : '语义分析：已关闭，使用原问题检索')
    : '';
  if (!evidence.length) {
    list.innerHTML = `${rawErrorText}<div class="card empty-state">没有命中证据。请先确认文档状态为 ready，或换一个更具体的问题。</div>`;
    return;
  }
  list.innerHTML = `${rawErrorText}${evidence.map((item) => {
    const chunk = item.chunk || {};
    const spans = chunk.source_spans || [];
    const locator = spans[0]?.locator || {};
    const kindTag = chunk.source_kind === 'code' ? ' <span class="source-tag">代码</span>' : '';
    return `<details id="evidence-${escapeHtml(item.evidence_id)}" class="card evidence-card"${expanded.has(`evidence-${item.evidence_id}`) ? ' open' : ''}>
      <summary class="evidence-card-header evidence-summary"><span><span class="evidence-id">${escapeHtml(item.evidence_id || 'E?')}</span> · ${escapeHtml(chunk.title || '未命名文档')}${kindTag}</span><span class="score">score ${Number(item.score ?? 0).toFixed(3)}</span></summary>
      <div class="evidence-content"><p class="evidence-highlight-note">标色为问题关键词匹配</p><p class="evidence-text">${highlightEvidence(chunk.text, state.searchQuestion)}</p>
      <div class="evidence-meta"><span>知识库：${escapeHtml(shortId(chunk.kb_id))}</span><span>文档：${escapeHtml(chunk.title || '-')}</span><span>document: ${escapeHtml(shortId(chunk.document_id))}</span><span>version: ${escapeHtml(shortId(chunk.version_id))}</span><span>chunk: ${escapeHtml(shortId(chunk.chunk_id))}</span><span>state: ${escapeHtml(chunk.search_state || '-')}</span></div>
      <div class="evidence-details"><span>来源定位：${escapeHtml(locator.kind || 'text')} · 第 ${escapeHtml(locator.page ?? locator.line_start ?? 1)} ${locator.page != null ? '页' : '行'} · 字符 ${escapeHtml(spans[0]?.text_start ?? 0)}-${escapeHtml(spans[0]?.text_end ?? 0)} · block ${escapeHtml(shortId(spans[0]?.block_id))}</span><button type="button" class="button secondary reader-open" data-document-id="${escapeHtml(chunk.document_id || '')}" data-title="${escapeHtml(chunk.title || '')}" data-format="${escapeHtml(locator.kind || 'text')}" data-page="${escapeHtml(locator.page ?? '')}" data-line="${escapeHtml(locator.line_start ?? '')}" data-line-end="${escapeHtml(locator.line_end ?? '')}" data-start="${escapeHtml(spans[0]?.text_start ?? 0)}" data-end="${escapeHtml(spans[0]?.text_end ?? 0)}">打开阅读器并定位到此位置</button></div></div>
    </details>`;
  }).join('')}`;
}

function renderRawErrorText(value) {
  const rawErrorText = typeof value === 'string' ? value.trim().slice(0, 4000) : '';
  if (!rawErrorText) return '';
  return `<details class="raw-error-output" open><summary>rawErrorText（AI 原始返回，trim 后）</summary><pre>${escapeHtml(rawErrorText)}</pre></details>`;
}

function readerTarget(button) {
  return {documentId: button.dataset.documentId, title: button.dataset.title, format: button.dataset.format,
    page: Number(button.dataset.page || 0), line: Number(button.dataset.line || 0), lineEnd: Number(button.dataset.lineEnd || 0),
    start: Number(button.dataset.start || 0), end: Number(button.dataset.end || 0)};
}

function setReaderOpen(open) {
  $('#reader-modal').classList.toggle('hidden', !open);
  document.body.classList.toggle('reader-open', open);
  if (!open) $('#reader-body').innerHTML = '';
}

function readerLocation(target) {
  const pageBased = target.page > 0 || target.format === 'pdf' || target.format === 'pptx';
  return `${pageBased ? `第 ${target.page || 1} 页` : `第 ${target.line || 1} 行`} · 字符 ${target.start}-${target.end}`;
}

async function sourceAvailable(url) {
  try {
    const response = await fetch(apiUrl(url), {headers: {Range: 'bytes=0-0'}});
    return response.ok || response.status === 206;
  } catch (error) {
    return false;
  }
}

async function openReader(button) {
  return openReaderTarget(readerTarget(button));
}

function openDocumentPreview(documentId, title, format) {
  return openReaderTarget({documentId, title, format: format || 'text', page: 1, line: 1, start: 0, end: 0, preview: true});
}

function lineOffset(text, line) {
  if (!line || line < 1) return 0;
  let offset = 0;
  for (let index = 1; index < line; index += 1) {
    const next = text.indexOf('\n', offset);
    if (next < 0) return text.length;
    offset = next + 1;
  }
  return offset;
}

async function openReaderTarget(target) {
  $('#reader-title').textContent = target.title || '文档阅读器';
  $('#reader-location').textContent = target.preview ? '全文预览' : readerLocation(target);
  $('#reader-status').textContent = '正在打开文档…';
  $('#reader-body').innerHTML = '';
  setReaderOpen(true);
  const contentUrl = kbPath(`/documents/${encodeURIComponent(target.documentId)}/content`);
  if (target.format === 'pdf' && await sourceAvailable(contentUrl)) {
    const frame = document.createElement('iframe');
    frame.className = 'reader-pdf';
    frame.title = target.title || 'PDF 阅读器';
    frame.src = `${apiUrl(contentUrl)}#page=${target.page || 1}`;
    $('#reader-body').appendChild(frame);
    $('#reader-status').textContent = '已定位到 PDF 页码；浏览器 PDF 阅读器负责页面内显示。';
    return;
  }
  try {
    const result = await apiFetch(kbPath(`/documents/${encodeURIComponent(target.documentId)}/reader-text`));
    const pre = document.createElement('pre');
    pre.className = 'reader-text';
    pre.textContent = result.text || '';
    $('#reader-body').appendChild(pre);
    const maxStart = Math.max(0, pre.textContent.length - 1);
    let start = Math.min(target.start, maxStart);
    let end = Math.min(Math.max(start + 1, target.end), pre.textContent.length);
    if (target.format === 'text' && target.line > 0 && !target.preview) {
      // Text/code chunks carry line locators; char offsets are chunk-relative,
      // so resolve the absolute offset from the line range instead.
      start = lineOffset(pre.textContent, target.line);
      const lastLine = Math.max(target.line, target.lineEnd || target.line);
      const nextLineStart = lineOffset(pre.textContent, lastLine + 1);
      end = nextLineStart >= pre.textContent.length
        ? pre.textContent.length
        : Math.max(start + 1, nextLineStart - 1);
    }
    if (pre.textContent && result.source_available !== false && target.format !== 'pdf' && !target.preview) {
      const range = document.createRange();
      const textNode = pre.firstChild;
      range.setStart(textNode, start);
      range.setEnd(textNode, Math.min(Math.max(start + 1, end), textNode.length));
      const mark = document.createElement('mark');
      mark.className = 'reader-target';
      range.surroundContents(mark);
      mark.scrollIntoView({block: 'center'});
    }
    if (result.source_available === false) {
      $('#reader-status').textContent = `原始文件不可用，显示入库时抽取的文本${target.preview ? '。' : `（定位可能不精确）。${readerLocation(target)}`}`;
    } else {
      $('#reader-status').textContent = target.preview ? '已打开文档预览。' : `已定位到 ${readerLocation(target)}。`;
    }
  } catch (error) {
    $('#reader-status').textContent = `阅读器打开失败：${error.message}`;
  }
}

$('#evidence-list').addEventListener('click', (event) => {
  const button = event.target.closest('.reader-open');
  if (button) { event.preventDefault(); openReader(button); }
});
$('#reader-modal').addEventListener('click', (event) => {
  if (event.target.closest('[data-reader-close]')) setReaderOpen(false);
});
document.addEventListener('keydown', (event) => { if (event.key === 'Escape' && !$('#reader-modal').classList.contains('hidden')) setReaderOpen(false); });

function renderAnswerPending(mode, style = 'conservative') {
  state.answerPending = true;
  state.lastPreviewText = '';
  const panel = $('#answer-panel');
  panel.className = 'card answer-card answer-loading';
  const aiLabel = state.serviceInfo ? `${state.serviceInfo.ai_provider} · ${state.serviceInfo.ai_model}` : 'AI 主引擎';
  const divergent = style === 'divergent';
  panel.innerHTML = `<div class="answer-head"><h3>回答</h3><div class="answer-status"><span class="status-tag processing">${divergent ? '发散生成中' : '生成中'}</span><span class="score">${escapeHtml(mode === 'grounded' ? aiLabel : 'extractive')}</span></div></div><p class="answer-body">${divergent ? '正在依据证据做推断与延伸，推演条目仍会逐条校验引用…' : '正在依据证据组织回答…'}</p><section class="answer-thinking"><strong>思考过程 · 实时回答草稿（尚未校验）</strong><div id="answer-live-preview" role="log" aria-live="polite">等待模型输出…</div></section>`;
}

function streamAnswerFailure(value, prefix = '') {
  const error = new Error(`${prefix}${value?.message || '流式生成失败。'}`);
  error.rawErrorOutput = typeof value?.rawErrorOutput === 'string' ? value.rawErrorOutput.trim().slice(0, 4000) : '';
  return error;
}

function structuredOutputHint() {
  if (!state.serviceInfo?.generation_enabled) return '';
  const mode = state.serviceInfo?.structured_mode === 'json' ? 'json' : 'tool';
  const next = mode === 'tool' ? 'json' : 'tool';
  const label = mode === 'tool' ? '工具调用' : 'JSON 模式';
  const nextLabel = next === 'tool' ? '工具调用' : 'JSON 模式';
  return `<div class="structured-hint"><p>当前结构化输出方式：<strong>${label}</strong>。若反复出现“JSON 字段不符合回答结构”，可切换到<strong>${nextLabel}</strong>后重试。</p><button type="button" class="button secondary xs" data-switch-structured="${next}">切换到${nextLabel}</button><span class="form-feedback" role="status"></span></div>`;
}

function renderAnswerFailure(error) {
  state.answerPending = false;
  const panel = $('#answer-panel');
  const preview = state.lastPreviewText || '';
  const rawErrorOutput = typeof error.rawErrorOutput === 'string' ? error.rawErrorOutput.trim().slice(0, 4000) : '';
  const rawErrorText = renderRawErrorText(state.lastSearch?.raw_error_text);
  panel.className = 'card answer-card answer-failed';
  panel.innerHTML = `<div class="answer-head"><h3>回答</h3><div class="answer-status"><span class="status-tag failed">生成失败</span></div></div><p class="answer-body">${escapeHtml(error.message || '生成服务暂时不可用。')}<br>证据仍然可用，请查看下方匹配结果。</p>${structuredOutputHint()}${rawErrorOutput ? `<details class="raw-error-output" open><summary>rawErrorOutput（AI 原始返回，trim 后）</summary><pre>${escapeHtml(rawErrorOutput)}</pre></details>` : ''}${rawErrorText}${preview ? `<section class="answer-thinking"><strong>未校验的实时草稿</strong><div id="answer-live-preview" class="answer-markdown" role="log">${AnswerMarkdown.render(preview)}</div></section>` : ''}`;
}

function renderAnswerPreview(text) {
  const preview = $('#answer-live-preview');
  if (!preview) return;
  state.lastPreviewText = text || '';
  // 草稿也走 Markdown 渲染：模型边输出边排版，避免用户先看到 ** 和 ##。
  preview.classList.add('answer-markdown');
  if (text) preview.innerHTML = AnswerMarkdown.render(text);
  else preview.textContent = '等待模型输出…';
  preview.scrollTop = preview.scrollHeight;
}

async function streamAnswer(payload, options = {}) {
  const onPreview = typeof options.onPreview === 'function' ? options.onPreview : renderAnswerPreview;
  const headers = new Headers({'Content-Type': 'application/json', 'Accept': 'text/event-stream'});
  const userId = localUserId();
  if (userId) headers.set('X-KB-User-ID', userId);
  const response = await fetch(apiUrl(kbPath('/answers/stream')), {
    method: 'POST',
    headers,
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const raw = await response.text();
    let detail = null;
    try { detail = JSON.parse(raw)?.detail; } catch { /* retain plain error */ }
    const message = typeof detail === 'object' && detail !== null ? (detail.message || JSON.stringify(detail)) : (detail || raw || response.statusText);
    throw streamAnswerFailure({message, rawErrorOutput: detail?.rawErrorOutput}, `${response.status} `);
  }
  if (!response.body) throw new Error('浏览器未提供流式响应读取能力。');

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let completed = null;
  const dispatch = (frame) => {
    let eventName = 'message';
    const data = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) eventName = line.slice(6).trim();
      else if (line.startsWith('data:')) data.push(line.slice(5).trimStart());
    }
    if (!data.length) return;
    const value = JSON.parse(data.join('\n'));
    if (eventName === 'preview') onPreview(value.text);
    else if (eventName === 'complete') completed = value;
    else if (eventName === 'error') throw streamAnswerFailure(value);
  };

  while (true) {
    const {value, done} = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), {stream: !done}).replace(/\r\n/g, '\n');
    const frames = buffer.split('\n\n');
    buffer = frames.pop() || '';
    frames.forEach(dispatch);
    if (done) break;
  }
  if (buffer.trim()) dispatch(buffer);
  if (!completed) throw new Error('流式连接结束，但没有收到经过校验的最终答案。');
  return completed;
}

function matchesAnswerSource(item, source) {
  if (source.chunk_id) return item.chunk?.chunk_id === source.chunk_id && item.chunk?.version_id === source.version_id;
  return item.chunk?.title === source.title && item.chunk?.text === source.quote;
}

function renderAnswerMarkdown(answer) {
  const sources = new Map((answer.sources || []).map((source) => [source.id, source]));
  return AnswerMarkdown.render(answer.answer || '', {
    renderCitation: (id) => {
      const source = sources.get(id);
      if (!source) return `[${escapeHtml(id)}]`;
      const evidence = state.lastSearch?.evidence?.find((item) => matchesAnswerSource(item, source));
      if (!evidence) return `[${escapeHtml(id)}]`;
      const target = `evidence-${evidence.evidence_id}`;
      const title = `文档：${source.title || '未命名文档'}`;
      return `<sup class="answer-citation"><a href="#${escapeHtml(target)}" title="${escapeHtml(title)}" aria-label="${escapeHtml(`${id}，${title}`)}">[${escapeHtml(id)}]</a></sup>`;
    },
  });
}

const FEEDBACK_REASONS = {irrelevant: '答非所问', incorrect: '事实错误', incomplete: '内容遗漏', citation: '引用不符', other: '其他'};

function answerFeedbackMarkup(answerId) {
  if (!answerId) return '';
  return `<div class="answer-feedback" data-answer-id="${escapeHtml(answerId)}">
      <div class="feedback-actions"><button type="button" data-rating="positive" aria-label="点赞">👍 点赞</button><button type="button" data-rating="negative" aria-expanded="false">🙅 错误</button></div>
      <div class="feedback-reasons" hidden>${Object.entries(FEEDBACK_REASONS).map(([key, label]) => `<button type="button" data-category="${key}">${label}</button>`).join('')}</div>
      <span class="feedback-message" role="status" aria-live="polite"></span>
    </div>`;
}

function renderAnswer(answer) {
  state.answerPending = Boolean(answer);
  state.lastAnswer = answer;
  const panel = $('#answer-panel');
  if (!answer) { state.answerPending = false; panel.className = 'hidden'; return; }
  // Keep one evidence list, including sources absent from the separate search response.
  const result = {...(state.lastSearch || {}), evidence: [...(state.lastSearch?.evidence || [])]};
  for (const source of answer.sources || []) {
    if (result.evidence.some((item) => matchesAnswerSource(item, source))) continue;
    let number = 1;
    while (result.evidence.some((item) => item.evidence_id === `E${number}`)) number++;
    result.evidence.push({evidence_id: `E${number}`, chunk: {...source, text: source.quote, source_spans: source.locator ? [{locator: source.locator}] : []}});
  }
  renderEvidence(result);
  state.answerPending = false;
  const claimKindLabels = {fact: '依据', inference: '推断', analogy: '类比', implication: '影响', hypothesis: '假设'};
  const claims = (answer.claims || []).map((claim) => {
    const kind = claim.kind || 'fact';
    const badge = kind === 'fact' ? '' : `<span class="claim-kind">${escapeHtml(claimKindLabels[kind] || '推断')}</span> `;
    return `<div class="claim-item">${badge}<strong>${escapeHtml(claim.id)}</strong> ${escapeHtml(claim.text)} <span class="mono">来源：${escapeHtml((claim.source_ids || []).join(', ') || '-')}</span></div>`;
  }).join('') || '<div class="claim-item">无结构化 Claim</div>';
  const followups = (answer.suggested_actions || []).filter((item) => String(item || '').trim());
  const followupBlock = followups.length
    ? `<div class="followups"><h4>可以继续追问</h4>${followups.map((item) => `<button type="button" class="followup-item" data-followup="${escapeHtml(item)}">${escapeHtml(item)}</button>`).join('')}</div>`
    : '';
  const warnings = (answer.warnings || []).map((warning) => `<div class="warning-text">${escapeHtml(warning)}</div>`).join('');
  const rawErrorText = renderRawErrorText(state.lastSearch?.raw_error_text);
  const styleLabel = answer.generation_style === 'divergent' ? '发散型 · ' : '';
  panel.className = 'card answer-card';
  panel.innerHTML = `<div class="answer-head"><h3>回答</h3><div class="answer-status"><span class="status-tag ${escapeHtml(answer.status || '')}">${escapeHtml(answer.status || '-')}</span><span class="score">${escapeHtml(styleLabel + (answer.verification_level || '-'))}</span></div></div>
    <div class="answer-body answer-markdown">${renderAnswerMarkdown(answer)}</div>
    ${answerFeedbackMarkup(answer.answer_id)}
    <div class="answer-metrics"><div class="metric"><span class="metric-label">confidence</span><span class="metric-value">${(Number(answer.confidence || 0) * 100).toFixed(0)}%</span></div><div class="metric"><span class="metric-label">confidence kind</span><span class="metric-value">${escapeHtml(answer.confidence_kind || '-')}</span></div><div class="metric"><span class="metric-label">request ID</span><span class="metric-value mono">${escapeHtml(shortId(answer.request_id))}</span></div><div class="metric"><span class="metric-label">answer ID</span><span class="metric-value mono">${escapeHtml(shortId(answer.answer_id))}</span></div></div>
    <div class="claims"><h4>Claims</h4>${claims}</div>${followupBlock}${warnings || rawErrorText ? `<div class="warnings"><h4>Warnings</h4>${warnings}${rawErrorText}</div>` : ''}`;
}

$('#answer-panel').addEventListener('click', (event) => {
  const feedbackButton = event.target.closest('.answer-feedback button');
  if (feedbackButton) { handleFeedback(feedbackButton); return; }
  const followup = event.target.closest('.followup-item');
  if (followup) {
    const input = $('#question-input');
    input.value = followup.dataset.followup || '';
    input.focus();
    input.scrollIntoView({block: 'center'});
    return;
  }
  const link = event.target.closest('.answer-citation a');
  if (!link) return;
  const target = document.getElementById(link.getAttribute('href').slice(1));
  if (!target) return;
  if (target.tagName === 'DETAILS') target.open = true;
  target.setAttribute('tabindex', '-1');
  target.focus({preventScroll: true});
});

function feedbackId() {
  // getRandomValues also works on the HTTP-only LAN console.
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  bytes[6] = (bytes[6] & 15) | 64;
  bytes[8] = (bytes[8] & 63) | 128;
  const hex = [...bytes].map((value) => value.toString(16).padStart(2, '0')).join('');
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

async function handleFeedback(button) {
  const box = button.closest('.answer-feedback');
  if (box.dataset.busy || box.dataset.saved) return;
  const reasons = box.querySelector('.feedback-reasons');
  if (button.dataset.rating === 'negative') {
    reasons.hidden = !reasons.hidden;
    button.setAttribute('aria-expanded', String(!reasons.hidden));
    return;
  }
  const message = box.querySelector('.feedback-message');
  // Keep exactly the same payload on retry after an uncertain network failure.
  const payload = box.feedbackPayload || {
    feedback_id: feedbackId(),
    rating: button.dataset.category ? 'negative' : 'positive',
    category: button.dataset.category || null,
  };
  box.feedbackPayload = payload;
  box.dataset.busy = 'true';
  box.querySelectorAll('button').forEach((item) => { item.disabled = true; });
  message.textContent = '正在保存…';
  try {
    await apiFetch(kbPath(`/answers/${encodeURIComponent(box.dataset.answerId)}/feedback`), {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload),
    });
    box.dataset.saved = 'true';
    reasons.hidden = true;
    message.textContent = '已保存，感谢反馈。将用于后续人工复核。';
  } catch (error) {
    message.textContent = error.message.startsWith('404')
      ? '该答案尚无存档，请重新生成后评价。'
      : '未确认保存成功，请点击重试（不会重复记录）。';
    button.textContent = '重试保存';
    button.disabled = false;
  } finally {
    delete box.dataset.busy;
  }
}

function iterativeEnabled() {
  const box = $('#iterative-retrieval-input');
  return box ? box.checked : true;
}

function collapseProgress(element = $('#search-progress')) {
  if (!element) return;
  element.classList.add('collapsed');
  const button = element.querySelector('[data-progress-toggle]');
  if (button) {
    button.textContent = '展开';
    button.setAttribute('aria-expanded', 'false');
  }
}

const progressStageLabels = {
  planning: '语义分析',
  retrieval: '检索',
  rerank: '证据重排',
  expansion: '证据补全',
  generation: '回答生成',
  done: '完成',
};

const PROGRESS_DURATION_THRESHOLD_SECONDS = 15;

function progressStepMessage(item) {
  if (item.status !== 'complete') return item.message;
  const completedMessages = {'基于证据生成回答中': '已完成生成', '基于证据生成发散型回答中': '已完成发散生成'};
  return completedMessages[item.message] || item.message;
}

function renderProgressTimeline(events, collapsed = false, element = $('#search-progress')) {
  if (!element) return;
  element.className = `search-progress${collapsed ? ' collapsed' : ''}`;
  const currentMessage = progressStepMessage(events.at(-1) || {}) || '准备开始…';
  const now = Date.now();
  element.innerHTML = `<div class="progress-header"><strong>检索过程</strong><span class="progress-current">${escapeHtml(currentMessage)}</span><button type="button" class="progress-toggle" data-progress-toggle aria-expanded="${!collapsed}">${collapsed ? '展开' : '收起'}</button></div><ol class="progress-timeline" aria-live="polite">${events.map((item, index) => {
    const status = item.status === 'active' ? 'active' : 'complete';
    const label = progressStageLabels[item.stage] || '处理中';
    const marker = status === 'active' ? '<span class="progress-spinner" aria-hidden="true"></span>' : '<span aria-hidden="true">✓</span>';
    const timestamp = Number(item.timestamp_ms) || now;
    const nextTimestamp = Number(events[index + 1]?.timestamp_ms) || now;
    const elapsedSeconds = Math.max(0, Math.floor((nextTimestamp - timestamp) / 1000));
    const clock = new Date(timestamp).toLocaleTimeString('zh-CN', {hour12: false});
    const duration = elapsedSeconds >= PROGRESS_DURATION_THRESHOLD_SECONDS ? ` · 耗时${elapsedSeconds}秒` : '';
    return `<li class="progress-step ${status}"><span class="progress-step-marker">${marker}</span><div class="progress-step-content"><strong>${escapeHtml(label)}</strong><span>${escapeHtml(progressStepMessage(item))}</span><small class="progress-step-meta">${escapeHtml(clock)}${duration}</small></div></li>`;
  }).join('')}</ol>`;
  element.querySelector('[data-progress-toggle]')?.addEventListener('click', () => {
    const isCollapsed = element.classList.toggle('collapsed');
    const button = element.querySelector('[data-progress-toggle]');
    if (button) {
      button.textContent = isCollapsed ? '展开' : '收起';
      button.setAttribute('aria-expanded', String(!isCollapsed));
    }
  });
}

function watchProgress(progressId, options = {}) {
  const element = Object.prototype.hasOwnProperty.call(options, 'element') ? options.element : $('#search-progress');
  const onResult = Object.prototype.hasOwnProperty.call(options, 'onResult') ? options.onResult : renderEvidence;
  const onEvent = typeof options.onEvent === 'function' ? options.onEvent : null;
  const isCollapsed = () => element?.classList.contains('collapsed') ?? false;
  let stopped = false;
  let lastRevision = 0;
  let lastResultRevision = 0;
  let lastEventRevision = 0;
  const events = [];
  const stream = new EventSource(apiUrl(kbPath(`/progress/${encodeURIComponent(progressId)}/stream`)));
  const refreshTimeline = () => renderProgressTimeline(events, isCollapsed(), element);
  const durationTimer = window.setInterval(() => {
    if (!stopped && events.length) refreshTimeline();
  }, 1000);
  renderProgressTimeline(events, false, element);

  const appendProgressEvent = (info) => {
    if (!info?.message || (info.revision && info.revision <= lastEventRevision)) return;
    events.forEach((item) => { item.status = 'complete'; });
    const stage = info.stage || 'processing';
    events.push({stage, message: info.message, timestamp_ms: Number(info.timestamp_ms) || Date.now(), status: stage === 'done' ? 'complete' : 'active'});
    lastEventRevision = info.revision || lastEventRevision;
    renderProgressTimeline(events, isCollapsed(), element);
    if (onEvent) onEvent(events, info);
  };

  const appendProgressHistory = (history, fallback) => {
    const pending = Array.isArray(history) ? history : [fallback];
    pending.filter((item) => item?.revision > lastEventRevision).forEach(appendProgressEvent);
  };

  stream.addEventListener('progress', (event) => {
    if (stopped) return;
    try {
      const info = JSON.parse(event.data);
      if (info?.revision && info.revision > lastRevision) {
        lastRevision = info.revision;
        appendProgressHistory(info.history, info);
        if (info.result && info.result_revision > lastResultRevision) {
          lastResultRevision = info.result_revision;
          const result = {
            ...info.result,
            query_analysis: info.query_analysis || info.result.query_analysis || state.lastSearch?.query_analysis,
          };
          if (typeof onResult === 'function') onResult(result);
        }
        if (info.stage === 'error') {
          state.lastProgressError = {message: info.message || '', rawErrorOutput: info.rawErrorOutput || ''};
        }
        if (info.stage === 'done' || info.stage === 'error') stream.close();
      }
    } catch { /* progress is best-effort */ }
  });
  return () => {
    stopped = true;
    stream.close();
    window.clearInterval(durationTimer);
    // The request has finished: the stream may have missed the final "done"
    // event, so fetch progress once more, then settle every stage locally
    // instead of leaving a spinner or an in-progress message behind.
    void (async () => {
      try {
        const info = await apiFetch(kbPath(`/progress/${encodeURIComponent(progressId)}`));
        if (info && info.revision && info.revision !== lastRevision) {
          lastRevision = info.revision;
          appendProgressHistory(info.history, info);
        }
      } catch { /* progress is best-effort */ }
      if (events.length) {
        events.forEach((item) => { item.status = 'complete'; });
        renderProgressTimeline(events, isCollapsed(), element);
        if (onEvent) onEvent(events, null);
      }
    })();
  };
}

function newProgressId() {
  if (window.crypto?.randomUUID) return window.crypto.randomUUID();
  const bytes = new Uint8Array(16);
  if (window.crypto?.getRandomValues) window.crypto.getRandomValues(bytes);
  else for (let i = 0; i < bytes.length; i += 1) bytes[i] = Math.floor(Math.random() * 256);
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hex = [...bytes].map((byte) => byte.toString(16).padStart(2, '0')).join('');
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

function selectedSourceKinds() {
  const kinds = [];
  if ($('#source-document-input')?.checked) kinds.push('document');
  if ($('#source-code-input')?.checked) kinds.push('code');
  return kinds;
}

function queryPayload() {
  const question = $('#question-input').value.trim();
  const limit = 5;
  if (!question) throw new Error('请先输入问题。');
  const sourceKinds = selectedSourceKinds();
  if (!sourceKinds.length) throw new Error('请至少勾选一种数据源：根据文档回答或根据代码回答。');
  return {
    question, limit, kb_ids: state.activeKnowledgeBaseId ? [state.activeKnowledgeBaseId] : [],
    auto_semantic_analysis: $('#semantic-analysis-input').checked,
    iterative_retrieval: iterativeEnabled(),
    source_kinds: sourceKinds,
    progress_id: newProgressId(),
  };
}

async function runSearch(withAnswer, style = 'conservative') {
  const divergent = style === 'divergent';
  const button = withAnswer
    ? (divergent ? $('#divergent-button') : $('#answer-button'))
    : $('#search-button');
  hideAlert('search-alert');
  let stopProgress = null;
  try {
    const payload = queryPayload();
    state.lastProgressError = null;
    stopProgress = watchProgress(payload.progress_id);
    state.searchQuestion = payload.question;
    renderEvidencePending();
    setBusy(button, true, withAnswer ? (divergent ? '发散生成中…' : '生成中…') : '检索中…');
    const searchRequest = {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)};
    if (!withAnswer) {
      renderAnswer(null);
      renderEvidence(await apiFetch(kbPath('/search'), searchRequest));
      return;
    }

    const mode = 'grounded';
    renderAnswerPending(mode, style);
    try {
      const searchResult = await apiFetch(kbPath('/search'), searchRequest);
      renderEvidence(searchResult);
      // Reuse the server-produced analysis so search and answer have one
      // auditable retrieval plan and do not call the AI planner twice.
      const answer = await streamAnswer({...payload, mode, generation_style: style, query_analysis: searchResult.query_analysis});
      renderAnswer(answer);
    } catch (error) {
      if (state.lastSearch) {
        let failure = error;
        if (!failure.rawErrorOutput) {
          let rawErrorOutput = state.lastProgressError?.rawErrorOutput || '';
          try {
            const progress = await apiFetch(kbPath(`/progress/${encodeURIComponent(payload.progress_id)}`));
            rawErrorOutput = progress.rawErrorOutput || rawErrorOutput;
          } catch { /* the answer stream error remains authoritative */ }
          if (rawErrorOutput) failure = streamAnswerFailure({message: error.message, rawErrorOutput});
        }
        renderAnswerFailure(failure);
        showAlert('search-alert', `回答生成失败，但证据仍已保留：${failure.message}`, 'error');
      } else {
        renderAnswer(null);
        showAlert('search-alert', `证据检索失败：${error.message}`, 'error');
      }
    }
  } catch (error) {
    showAlert('search-alert', error.message, 'error');
  } finally {
    setBusy(button, false);
    if (stopProgress) stopProgress();
    collapseProgress();
  }
}

// ---------------------------------------------------------------------------
// 问答：历史会话 + 多轮上下文
// ---------------------------------------------------------------------------

function conversationTime(value) {
  if (!value) return '';
  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) return '';
  const diff = Date.now() - date.valueOf();
  if (diff < 60 * 1000) return '刚刚';
  if (diff < 60 * 60 * 1000) return `${Math.floor(diff / 60000)} 分钟前`;
  if (diff < 24 * 60 * 60 * 1000) return `${Math.floor(diff / 3600000)} 小时前`;
  return date.toLocaleDateString('zh-CN');
}

function chatMessageFromServer(message) {
  const metadata = message.metadata || {};
  const answer = message.role === 'assistant'
    ? {
      answer: message.content,
      answer_id: message.answer_id || null,
      status: metadata.status || 'answered',
      confidence: metadata.confidence,
      verification_level: metadata.verification_level,
      generation_style: metadata.generation_style || 'conservative',
      claims: metadata.claims || [],
      sources: metadata.sources || [],
      warnings: metadata.warnings || [],
      suggested_actions: metadata.suggested_actions || [],
      query_analysis: metadata.query_analysis || null,
    }
    : null;
  return {
    key: `server-${message.message_id}`,
    role: message.role,
    content: message.content || '',
    answer,
    answerId: message.answer_id || null,
    progressEvents: (metadata.progress_events || []).map((item) => ({...item, status: 'complete'})),
    progressOpen: false,
    pending: false,
    error: null,
    rawErrorOutput: '',
  };
}

function chatProgressEvents(message) {
  return Array.isArray(message.progressEvents) ? message.progressEvents : [];
}

function chatProgressTimeline(message) {
  const events = chatProgressEvents(message);
  const live = Boolean(message.pending);
  const now = Date.now();
  return events.map((item, index) => {
    const status = live && item.status === 'active' && index === events.length - 1 ? 'active' : 'complete';
    const label = progressStageLabels[item.stage] || '处理中';
    const marker = status === 'active' ? '<span class="progress-spinner" aria-hidden="true"></span>' : '<span aria-hidden="true">✓</span>';
    const timestamp = Number(item.timestamp_ms) || now;
    const nextTimestamp = Number(events[index + 1]?.timestamp_ms) || now;
    const elapsedSeconds = Math.max(0, Math.floor((nextTimestamp - timestamp) / 1000));
    const clock = new Date(timestamp).toLocaleTimeString('zh-CN', {hour12: false});
    const duration = elapsedSeconds >= PROGRESS_DURATION_THRESHOLD_SECONDS ? ` · 耗时${elapsedSeconds}秒` : '';
    return `<li class="progress-step ${status}"><span class="progress-step-marker">${marker}</span><div class="progress-step-content"><strong>${escapeHtml(label)}</strong><span>${escapeHtml(progressStepMessage(item))}</span><small class="progress-step-meta">${escapeHtml(clock)}${duration}</small></div></li>`;
  }).join('');
}

function chatProgressInner(message) {
  const events = chatProgressEvents(message);
  const live = Boolean(message.pending);
  const current = events.length ? progressStepMessage(events.at(-1)) : (live ? '准备开始…' : '已完成');
  const count = events.length ? `<span class="chat-progress-count">${events.length} 步</span>` : '';
  return `<summary><span class="chat-progress-title">检索过程</span>${count}<span class="chat-progress-current">${escapeHtml(current || '已完成')}</span></summary><ol class="progress-timeline">${chatProgressTimeline(message)}</ol>`;
}

function chatProgressMarkup(message) {
  const events = chatProgressEvents(message);
  if (!events.length && !message.pending) return '';
  const open = Boolean(message.pending || message.progressOpen);
  return `<details class="chat-progress-block" data-progress-key="${escapeHtml(message.key)}"${open ? ' open' : ''}>${chatProgressInner(message)}</details>`;
}

function updateChatProgressDom(message) {
  const node = [...document.querySelectorAll('#chat-thread .chat-progress-block')]
    .find((item) => item.dataset.progressKey === message.key);
  if (!node) return;
  // 保留用户手动展开/收起的状态，只替换内容。
  const open = node.open;
  node.innerHTML = chatProgressInner(message);
  node.open = open;
}

function renderConversations() {
  const list = $('#conversation-list');
  if (!list) return;
  const counter = $('#conversation-count');
  if (counter) counter.textContent = `${state.chat.conversations.length} 个`;
  if (!state.chat.conversations.length) {
    list.innerHTML = '<p class="empty-cell">还没有会话记录。输入第一个问题就会创建会话。</p>';
    return;
  }
  list.innerHTML = state.chat.conversations.map((item) => {
    const active = item.conversation_id === state.chat.activeId;
    const title = item.title || item.last_question || '新会话';
    const meta = [item.message_count ? `${item.message_count} 条消息` : '空会话', conversationTime(item.updated_at)].filter(Boolean).join(' · ');
    return `<div class="conversation-item${active ? ' active' : ''}" data-conversation-id="${escapeHtml(item.conversation_id)}" role="button" tabindex="0" title="${escapeHtml(item.last_question || title)}">
      <span class="conversation-item-title">${escapeHtml(title)}</span>
      <span class="conversation-item-meta">${escapeHtml(meta)}</span>
      <span class="conversation-item-actions"><button type="button" data-conversation-action="rename" title="重命名会话">改名</button><button type="button" data-conversation-action="delete" title="删除会话">删除</button></span>
    </div>`;
  }).join('');
}

async function loadConversations() {
  const list = $('#conversation-list');
  if (!list) return;
  if (!state.activeKnowledgeBaseId) {
    state.chat.conversations = [];
    renderConversations();
    list.innerHTML = '<p class="empty-cell">没有可访问的知识库，无法读取会话。</p>';
    return;
  }
  try {
    const result = await apiFetch(kbPath('/conversations?limit=100'));
    state.chat.conversations = Array.isArray(result?.conversations) ? result.conversations : [];
  } catch (error) {
    state.chat.conversations = [];
    list.innerHTML = `<p class="empty-cell">会话读取失败：${escapeHtml(error.message)}</p>`;
    return;
  }
  renderConversations();
  if (state.chat.activeId) {
    const active = state.chat.conversations.find((item) => item.conversation_id === state.chat.activeId);
    if (active) {
      state.chat.activeTitle = active.title || state.chat.activeTitle;
      const titleNode = $('#chat-title');
      if (titleNode) titleNode.textContent = state.chat.activeTitle || '新会话';
    }
  }
}

function newConversationDisabled(chat = state.chat) {
  return Boolean(chat.sending) || (chat.messages || []).length === 0;
}

function updateNewConversationButton() {
  const button = $('#new-conversation');
  if (!button || button.dataset.busy) return;
  button.disabled = newConversationDisabled();
  button.title = button.disabled ? '当前已是空白会话' : '打开一个新的空会话';
}

function startNewChat() {
  if (state.chat.sending) return;
  if (state.chat.messages.length === 0) {
    $('#chat-input')?.focus();
    return;
  }
  const button = $('#new-conversation');
  setBusy(button, true, '创建中…');
  void (async () => {
    try {
      const summary = await createChatConversation('');
      state.chat.messages = [];
      state.chat.preview = '';
      state.chat.activeTitle = summary.title || '';
      renderConversations();
      renderChatThread();
      $('#chat-input')?.focus();
      await loadConversations();
    } catch (error) {
      showAlert('chat-alert', `新建会话失败：${error.message}`, 'error');
    } finally {
      setBusy(button, false);
      updateNewConversationButton();
    }
  })();
}

async function openChatConversation(conversationId) {
  if (!conversationId || state.chat.sending) return;
  hideAlert('chat-alert');
  try {
    const detail = await apiFetch(kbPath(`/conversations/${encodeURIComponent(conversationId)}`));
    state.chat.activeId = conversationId;
    state.chat.activeTitle = detail?.conversation?.title || '';
    state.chat.messages = (detail?.messages || []).map(chatMessageFromServer);
    state.chat.preview = '';
    renderConversations();
    renderChatThread();
  } catch (error) {
    showAlert('chat-alert', `会话打开失败：${error.message}`, 'error');
  }
}

async function createChatConversation(title) {
  const summary = await apiFetch(kbPath('/conversations'), {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({title: String(title || '').slice(0, 60)}),
  });
  state.chat.activeId = summary.conversation_id;
  state.chat.activeTitle = summary.title || title || '';
  state.chat.conversations = [summary, ...state.chat.conversations.filter((item) => item.conversation_id !== summary.conversation_id)];
  renderConversations();
  return summary;
}

function renderChatSource(source, messageKey, index) {
  const locator = source.locator || {};
  const kind = locator.kind || 'text';
  const page = locator.page ?? '';
  const line = locator.line_start ?? '';
  const target = `chat-source-${messageKey}-${index}`;
  const location = page ? `第 ${page} 页` : (line ? `第 ${line} 行` : '全文');
  return `<details id="${escapeHtml(target)}" class="chat-source">
    <summary><span class="evidence-id">${escapeHtml(source.id || `E${index + 1}`)}</span> · ${escapeHtml(source.title || '未命名文档')}</summary>
    <p class="evidence-text">${escapeHtml(source.quote || '')}</p>
    <div class="evidence-meta"><span>文档：${escapeHtml(source.title || '-')}</span><span>document: ${escapeHtml(shortId(source.document_id))}</span><span>version: ${escapeHtml(shortId(source.version_id))}</span><span>chunk: ${escapeHtml(shortId(source.chunk_id))}</span><span>定位：${escapeHtml(kind)} · ${escapeHtml(location)}</span></div>
    <div class="evidence-details"><button type="button" class="button secondary xs reader-open" data-document-id="${escapeHtml(source.document_id || '')}" data-title="${escapeHtml(source.title || '')}" data-format="${escapeHtml(kind)}" data-page="${escapeHtml(page)}" data-line="${escapeHtml(line)}" data-line-end="${escapeHtml(locator.line_end ?? '')}" data-start="0" data-end="0">打开阅读器并定位</button></div>
  </details>`;
}

function renderChatAnswerMarkdown(answer, messageKey) {
  const sources = new Map((answer.sources || []).map((source, index) => [source.id, {source, index}]));
  return AnswerMarkdown.render(answer.answer || '', {
    renderCitation: (id) => {
      const entry = sources.get(id);
      if (!entry) return `[${escapeHtml(id)}]`;
      const title = `文档：${entry.source.title || '未命名文档'}`;
      return `<sup class="answer-citation"><a href="#${escapeHtml(`chat-source-${messageKey}-${entry.index}`)}" title="${escapeHtml(title)}" aria-label="${escapeHtml(`${id}，${title}`)}">[${escapeHtml(id)}]</a></sup>`;
    },
  });
}

function renderChatMessage(message, index) {
  const key = message.key || `message-${index}`;
  if (message.role === 'user') {
    return `<article class="chat-message user" data-message-key="${escapeHtml(key)}"><div class="chat-avatar">你</div><div class="chat-bubble">${escapeHtml(message.content || '').replace(/\n/g, '<br>')}</div></article>`;
  }
  if (message.pending) {
    const preview = message.preview || '正在检索证据并组织回答…';
    return `<article class="chat-message assistant" data-message-key="${escapeHtml(key)}"><div class="chat-avatar">AI</div><div class="chat-bubble">${chatProgressMarkup(message)}<div class="chat-typing"><span class="progress-spinner" aria-hidden="true"></span><div class="chat-typing-text answer-markdown" id="chat-live-preview" role="log" aria-live="polite">${message.preview ? AnswerMarkdown.render(message.preview) : escapeHtml(preview)}</div></div></div></article>`;
  }
  if (message.error) {
    const raw = typeof message.rawErrorOutput === 'string' ? message.rawErrorOutput.trim().slice(0, 4000) : '';
    return `<article class="chat-message assistant" data-message-key="${escapeHtml(key)}"><div class="chat-avatar">AI</div><div class="chat-bubble">${chatProgressMarkup(message)}<p class="chat-error">${escapeHtml(message.error)}</p>${raw ? `<details class="raw-error-output" open><summary>rawErrorOutput（AI 原始返回，trim 后）</summary><pre>${escapeHtml(raw)}</pre></details>` : ''}<p class="chat-error-hint">本轮提问已写入会话，可直接重新提问；如反复失败，请到「配置」检查主 AI。</p></div></article>`;
  }
  const answer = message.answer || {};
  const sources = answer.sources || [];
  const warnings = (answer.warnings || []).filter((item) => String(item || '').trim());
  const followups = (answer.suggested_actions || []).filter((item) => String(item || '').trim());
  const status = answer.status || 'answered';
  const styleLabel = answer.generation_style === 'divergent' ? '发散型 · ' : '';
  const confidence = answer.confidence == null ? '' : `<span class="chat-message-meta">置信 ${(Number(answer.confidence) * 100).toFixed(0)}%</span>`;
  const sourcesBlock = sources.length
    ? `<details class="chat-sources"><summary>证据 ${sources.length} 条（回答中的 [E#] 可点击定位）</summary>${sources.map((source, sourceIndex) => renderChatSource(source, key, sourceIndex)).join('')}</details>`
    : '';
  const followupBlock = followups.length
    ? `<div class="followups"><h4>可以继续追问</h4>${followups.map((item) => `<button type="button" class="followup-item" data-followup="${escapeHtml(item)}">${escapeHtml(item)}</button>`).join('')}</div>`
    : '';
  const warningsBlock = warnings.length ? `<div class="warnings"><h4>Warnings</h4>${warnings.map((warning) => `<div class="warning-text">${escapeHtml(warning)}</div>`).join('')}</div>` : '';
  const resolved = (answer.query_analysis?.resolved_question || '').trim();
  const resolvedBlock = resolved && resolved !== (answer.query_analysis?.original_question || '').trim()
    ? `<p class="chat-resolved">追问已结合上下文改写为：${escapeHtml(resolved)}</p>`
    : '';
  return `<article class="chat-message assistant" data-message-key="${escapeHtml(key)}"><div class="chat-avatar">AI</div><div class="chat-bubble">
    <div class="chat-answer-head"><span class="status-tag ${escapeHtml(status)}">${escapeHtml(status)}</span><span class="score">${escapeHtml(styleLabel + (answer.verification_level || '-'))}</span>${confidence}</div>
    ${chatProgressMarkup(message)}
    <div class="answer-body answer-markdown">${renderChatAnswerMarkdown(answer, key)}</div>
    ${resolvedBlock}
    ${sourcesBlock}
    ${followupBlock}
    ${warningsBlock}
    ${answerFeedbackMarkup(message.answerId || answer.answer_id)}
  </div></article>`;
}

function renderChatThread() {
  const thread = $('#chat-thread');
  if (!thread) return;
  const titleNode = $('#chat-title');
  if (titleNode) titleNode.textContent = state.chat.activeTitle || '新会话';
  const subtitle = $('#chat-subtitle');
  if (subtitle) {
    subtitle.textContent = state.chat.activeId
      ? '同一会话内的追问会带上此前问答作为上下文；答案仍只引用本轮知识库证据。'
      : '在下方输入第一个问题；发送后会创建一个新会话。';
  }
  const hasActive = Boolean(state.chat.activeId);
  const renameButton = $('#rename-conversation');
  const deleteButton = $('#delete-conversation');
  if (renameButton) renameButton.hidden = !hasActive;
  if (deleteButton) deleteButton.hidden = !hasActive;
  updateNewConversationButton();
  if (!state.chat.messages.length) {
    thread.innerHTML = '<div class="chat-empty"><strong>开始一段可追溯的多轮问答</strong><p>左侧是历史会话，点击即可打开完整对话；下方输入问题会新建一个会话。</p><p class="chat-empty-tip">追问会自动带上本会话历史，用来消解“它 / 这个 / 上面说的”这类指代；事实仍然只能来自本轮检索到的知识库证据。</p></div>';
    return;
  }
  thread.innerHTML = state.chat.messages.map((message, index) => renderChatMessage(message, index)).join('');
  thread.scrollTop = thread.scrollHeight;
}

function chatSelectedSourceKinds() {
  const kinds = [];
  if ($('#chat-source-document')?.checked) kinds.push('document');
  if ($('#chat-source-code')?.checked) kinds.push('code');
  return kinds;
}

async function sendChatMessage() {
  if (state.chat.sending) return;
  hideAlert('chat-alert');
  const input = $('#chat-input');
  const question = (input?.value || '').trim();
  if (!question) { showAlert('chat-alert', '请先输入问题。', 'info'); return; }
  const sourceKinds = chatSelectedSourceKinds();
  if (!sourceKinds.length) { showAlert('chat-alert', '请至少选择一种数据源：文档或代码。', 'error'); return; }
  if (!state.activeKnowledgeBaseId) { showAlert('chat-alert', '当前没有可用的知识库，无法开始会话。', 'error'); return; }
  const divergent = Boolean($('#chat-divergent')?.checked);
  const sendButton = $('#chat-send');
  let stopProgress = null;
  state.chat.sending = true;
  setBusy(sendButton, true, '生成中…');
  input.value = '';
  try {
    if (!state.chat.activeId) await createChatConversation(question);
    const userMessage = {key: `local-user-${Date.now()}`, role: 'user', content: question, pending: false};
    const pendingMessage = {
      key: `local-answer-${Date.now()}`, role: 'assistant', pending: true, preview: '',
      progressEvents: [], progressOpen: true,
    };
    state.chat.messages.push(userMessage, pendingMessage);
    renderChatThread();
    const payload = {
      question,
      limit: 5,
      kb_ids: [state.activeKnowledgeBaseId],
      auto_semantic_analysis: true,
      iterative_retrieval: iterativeEnabled(),
      source_kinds: sourceKinds,
      progress_id: newProgressId(),
      conversation_id: state.chat.activeId,
      mode: 'grounded',
      generation_style: divergent ? 'divergent' : 'conservative',
    };
    stopProgress = watchProgress(payload.progress_id, {
      element: null,
      onResult: null,
      onEvent: (events) => {
        pendingMessage.progressEvents = events.map((item) => ({...item}));
        updateChatProgressDom(pendingMessage);
      },
    });
    try {
      const answer = await streamAnswer(payload, {
        onPreview: (text) => {
          pendingMessage.preview = text || '';
          const node = document.getElementById('chat-live-preview');
          if (node) {
            if (text) node.innerHTML = AnswerMarkdown.render(text);
            else node.textContent = '等待模型输出…';
            node.scrollTop = node.scrollHeight;
          }
          const thread = $('#chat-thread');
          if (thread) thread.scrollTop = thread.scrollHeight;
        },
      });
      pendingMessage.pending = false;
      pendingMessage.preview = '';
      pendingMessage.answer = answer;
      pendingMessage.answerId = answer.answer_id || null;
      pendingMessage.progressOpen = false;
      state.chat.preview = '';
      renderChatThread();
      await loadConversations();
    } catch (error) {
      pendingMessage.pending = false;
      pendingMessage.preview = '';
      pendingMessage.error = error.message || '生成失败';
      pendingMessage.rawErrorOutput = error.rawErrorOutput || '';
      pendingMessage.progressOpen = false;
      renderChatThread();
      showAlert('chat-alert', `本轮回答失败：${error.message}`, 'error');
      await loadConversations();
    } finally {
      if (stopProgress) stopProgress();
    }
  } catch (error) {
    // 会话创建失败：没有写入任何消息，恢复输入内容方便重试。
    showAlert('chat-alert', `会话创建失败：${error.message}`, 'error');
    if (input) input.value = question;
  } finally {
    state.chat.sending = false;
    setBusy(sendButton, false);
    updateNewConversationButton();
  }
}

async function renameChatConversation(conversationId) {
  const current = state.chat.conversations.find((item) => item.conversation_id === conversationId);
  const name = await requestTextInput({
    title: '重命名会话',
    description: '只修改会话标题，不影响已保存的问答记录与答案快照。',
    label: '会话标题',
    placeholder: current?.title || '例如：入库组件选型',
    submitLabel: '保存',
  });
  if (!name || !name.trim()) return;
  try {
    const summary = await apiFetch(kbPath(`/conversations/${encodeURIComponent(conversationId)}`), {
      method: 'PATCH', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({title: name.trim()}),
    });
    state.chat.conversations = state.chat.conversations.map((item) =>
      item.conversation_id === conversationId ? {...item, ...summary} : item);
    if (state.chat.activeId === conversationId) state.chat.activeTitle = summary.title || name.trim();
    renderConversations();
    renderChatThread();
  } catch (error) {
    showAlert('chat-alert', `重命名失败：${error.message}`, 'error');
  }
}

async function deleteChatConversation(conversationId) {
  if (state.chat.sending) {
    showAlert('chat-alert', '本轮回答还在生成中，完成后再删除会话。', 'info');
    return;
  }
  const current = state.chat.conversations.find((item) => item.conversation_id === conversationId);
  const title = current?.title || '新会话';
  if (!window.confirm(`确定删除会话“${title}”吗？\n会话中的问答记录会一并删除；答案快照与人工反馈评测记录会保留。`)) return;
  try {
    await apiFetch(kbPath(`/conversations/${encodeURIComponent(conversationId)}`), {method: 'DELETE'});
    state.chat.conversations = state.chat.conversations.filter((item) => item.conversation_id !== conversationId);
    if (state.chat.activeId === conversationId) {
      state.chat.activeId = null;
      state.chat.activeTitle = '';
      state.chat.messages = [];
    }
    renderConversations();
    renderChatThread();
  } catch (error) {
    showAlert('chat-alert', `删除失败：${error.message}`, 'error');
  }
}

function definitionList(element, values) {
  element.innerHTML = Object.entries(values).map(([label, value]) => `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd></div>`).join('');
}

function providerName(provider) {
  return provider === 'ollama' ? 'Ollama' : provider === 'openai' || provider === 'openai-compatible' || provider === 'openai_compatible' ? 'OpenAI 兼容' : provider || '未配置';
}

function populateProviderForm(group, config) {
  if (!config) return;
  $(`#${group}-provider-select`).value = config.provider || 'ollama';
  $(`#${group}-endpoint-input`).value = config.base_url || '';
  $(`#${group}-model-input`).value = config.model || '';
  $(`#${group}-api-key-input`).value = '';
  $(`#${group}-clear-key-input`).checked = false;
  $(`#${group}-api-key-state`).textContent = config.api_key_configured ? '（已保存；输入新 Key 可替换）' : '（未配置）';
  $(`#${group}-provider-badge`).textContent = providerName(config.provider);
  if (group === 'ai') {
    const modeSelect = $('#ai-structured-mode');
    if (modeSelect && config.structured_mode) modeSelect.value = config.structured_mode;
    state.serviceInfo = {...(state.serviceInfo || {}), structured_mode: config.structured_mode || state.serviceInfo?.structured_mode};
  }
}

function renderProviderPlans(data) {
  state.providerPlans = data?.plans || [];
  state.activeProviderPlanId = data?.active_plan_id || null;
  const activePlan = state.providerPlans.find((plan) => plan.active);
  $('#active-provider-plan').textContent = activePlan ? `当前：${activePlan.name}` : '自定义配置';
  const list = $('#provider-plan-list');
  list.textContent = '';
  if (!state.providerPlans.length) {
    const empty = document.createElement('p');
    empty.className = 'empty-cell';
    empty.textContent = '还没有已保存的方案。填写上方名称即可保存当前配置。';
    list.append(empty);
    return;
  }
  for (const plan of state.providerPlans) {
    const card = document.createElement('article');
    card.className = 'provider-plan-item';
    const heading = document.createElement('div');
    heading.className = 'provider-plan-heading';
    const name = document.createElement('h4');
    name.textContent = plan.name;
    heading.append(name);
    if (plan.active) {
      const active = document.createElement('span');
      active.className = 'provider-plan-active';
      active.textContent = '当前使用';
      heading.append(active);
    }
    const settings = document.createElement('div');
    settings.className = 'provider-plan-settings';
    for (const [group, title] of [['ai', '主 AI'], ['embedding', 'Embedding']]) {
      const config = plan.settings[group];
      const line = document.createElement('p');
      const label = document.createElement('strong');
      label.textContent = `${title}：`;
      const detail = document.createElement('span');
      detail.textContent = `${providerName(config.provider)} · ${config.model} · ${config.base_url}`;
      line.append(label, detail);
      settings.append(line);
    }
    const footer = document.createElement('div');
    footer.className = 'provider-plan-footer';
    const saved = document.createElement('span');
    saved.textContent = `保存于 ${formatDate(plan.created_at)}`;
    const actions = document.createElement('div');
    actions.className = 'provider-plan-actions';
    const activate = document.createElement('button');
    activate.className = 'button secondary xs';
    activate.type = 'button';
    activate.textContent = plan.active ? '正在使用' : '切换到此方案';
    activate.disabled = Boolean(plan.active);
    activate.addEventListener('click', () => activateProviderPlan(plan.id, activate));
    const remove = document.createElement('button');
    remove.className = 'button secondary xs';
    remove.type = 'button';
    remove.textContent = '删除';
    remove.addEventListener('click', () => deleteProviderPlan(plan, remove));
    actions.append(activate, remove);
    footer.append(saved, actions);
    card.append(heading, settings, footer);
    list.append(card);
  }
}

async function loadSettings() {
  try {
    const [info, health, providers, providerPlans] = await Promise.all([
      apiFetch('/service-info'),
      apiFetch('/health/ready'),
      apiFetch('/api/v1/settings/providers'),
      apiFetch('/api/v1/settings/provider-plans'),
    ]);
    state.serviceInfo = info; state.health = health;
    $('#pipeline-badge').className = 'badge ok';
    $('#pipeline-badge').textContent = health.pipeline_revision || 'ready';
    $('#sidebar-status-dot').className = 'status-dot ok';
    $('#sidebar-status').textContent = 'API 正常';
    $('#settings-api-status').className = 'status-pill ok';
    $('#settings-api-status').textContent = 'ready';
    definitionList($('#connection-list'), {'API 地址': baseUrl(), 'API 前缀': info.api || '/api/v1', '健康检查': info.health || '/health/ready', 'Swagger': `${baseUrl()}/docs`, '流水线版本': health.pipeline_revision || '-'});
    const swaggerLink = $('#swagger-link');
    if (swaggerLink) swaggerLink.href = `${baseUrl()}/docs`;
    const apiDocsLink = $('#api-docs-link');
    if (apiDocsLink) apiDocsLink.href = `${baseUrl()}/docs`;
    const apiInput = $('#api-base-input');
    if (apiInput && document.activeElement !== apiInput) apiInput.value = apiBase;
    const apiBadge = $('#current-api-base');
    if (apiBadge) apiBadge.textContent = apiBase || '同源（当前页面）';
    populateProviderForm('ai', providers.ai);
    populateProviderForm('embedding', providers.embedding);
    renderProviderPlans(providerPlans);
    $('#embedding-rebuild-state').textContent = providers.embedding_rebuild_running
      ? '状态：存量段落向量重建进行中。'
      : '';
    definitionList($('#model-list'), {'检索服务': info.model_service || 'BM25', '迭代补全': info.iterative_retrieval_enabled ? `默认开启（≤${info.iterative_max_rounds ?? 3} 轮）` : '默认关闭', 'Grounded 回答': info.generation_enabled ? '已启用' : '未配置', '搜索服务': 'Meilisearch（服务边界已运行）'});
    definitionList($('#ingestion-list'), {'最大上传': '50 MiB', '支持格式': 'PDF / DOCX / PPTX / TXT / MD / CSV', '分块器': 'SemanticChunker（Para-ID）', '任务方式': 'Redis/Celery + 独立 Worker', '索引状态': 'PostgreSQL 持久化，API 重启自动重建检索缓存'});
    hideAlert('settings-alert');
  } catch (error) {
    $('#pipeline-badge').className = 'badge error'; $('#pipeline-badge').textContent = '离线';
    $('#sidebar-status-dot').className = 'status-dot error'; $('#sidebar-status').textContent = 'API 不可用';
    $('#settings-api-status').className = 'status-pill error'; $('#settings-api-status').textContent = 'error';
    showAlert('settings-alert', `配置读取失败：${error.message}`, 'error');
  }
}

function diagnostics() {
  return JSON.stringify({api_base_url: baseUrl(), health: state.health, service: state.serviceInfo, model_backend: state.serviceInfo?.generation_enabled ? state.serviceInfo.ai_provider : 'disabled', embedding_backend: state.serviceInfo?.embedding_provider || 'disabled', search_backend: 'meilisearch', environment: 'compose'}, null, 2);
}

async function copyDiagnostics() {
  try { await navigator.clipboard.writeText(diagnostics()); showAlert('settings-alert', '诊断信息已复制。', 'success'); }
  catch { showAlert('settings-alert', diagnostics(), 'info'); }
}

async function clearKnowledgeBase() {
  const typed = await requestTextInput({
    title: '清空当前知识库',
    description: '此操作会删除当前知识库的文档、版本和切片，但保留答案快照与反馈记录。',
    label: '请输入“确认清空”以继续',
    placeholder: '确认清空',
    submitLabel: '继续清空',
  });
  if (typed !== '确认清空') {
    showAlert('settings-alert', '未完成清空：确认文本不匹配。', 'info');
    return;
  }
  const button = $('#clear-knowledge-base');
  setBusy(button, true, '清空中…');
  try {
    const result = await apiFetch(kbPath('/documents'), {
      method: 'DELETE',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({confirmation: 'CLEAR_KNOWLEDGE_BASE'}),
    });
    state.documents = [];
    state.documentsPage.offset = 0;
    state.lastSearch = null;
    state.lastAnswer = null;
    renderAnswer(null);
    renderEvidence({evidence: []});
    await Promise.all([loadDocuments(), loadGlossary()]);
    showAlert('settings-alert', '知识库已清空：' + result.documents + ' 份文档，' + result.files + ' 个待处理文件。答案评测记录已保留。', 'success');
  } catch (error) {
    showAlert('settings-alert', '清空失败：' + error.message, 'error');
  } finally {
    setBusy(button, false);
  }
}

async function saveProviderSettings(group) {
  const button = $(`#save-${group}-provider`);
  const feedback = $(`#${group}-provider-feedback`);
  const apiKey = $(`#${group}-api-key-input`).value;
  const clearApiKey = $(`#${group}-clear-key-input`).checked;
  setBusy(button, true, '正在应用…');
  feedback.textContent = '';
  feedback.className = 'form-feedback';
  try {
    const result = await apiFetch(`/api/v1/settings/providers/${group}`, {
      method: 'PUT',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        provider: $(`#${group}-provider-select`).value,
        base_url: $(`#${group}-endpoint-input`).value.trim(),
        model: $(`#${group}-model-input`).value.trim(),
        api_key: apiKey || null,
        clear_api_key: clearApiKey,
      }),
    });
    await loadSettings();
    const message = group === 'embedding' && result.embedding_rebuild_started
      ? '已切换。存量段落正在后台重新向量化，期间检索可能暂时退化为关键词召回。'
      : '已保存并切换，新请求将使用当前配置。';
    feedback.textContent = message;
    feedback.className = 'form-feedback success';
    showAlert('settings-alert', message, 'success');
  } catch (error) {
    feedback.textContent = `保存失败：${error.message}`;
    feedback.className = 'form-feedback error';
    showAlert('settings-alert', `配置切换失败：${error.message}`, 'error');
  } finally {
    $(`#${group}-api-key-input`).value = '';
    $(`#${group}-clear-key-input`).checked = false;
    setBusy(button, false);
  }
}

async function saveStructuredOutput(mode, button = null) {
  const target = button || $('#save-structured-output');
  const feedback = $('#structured-output-feedback');
  if (feedback) { feedback.textContent = ''; feedback.className = 'form-feedback'; }
  setBusy(target, true, '正在应用…');
  try {
    const result = await apiFetch('/api/v1/settings/structured-output', {
      method: 'PUT',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({mode}),
    });
    const applied = result.settings?.ai?.structured_mode || mode;
    state.serviceInfo = {...(state.serviceInfo || {}), structured_mode: applied};
    const select = $('#ai-structured-mode');
    if (select) select.value = applied;
    const message = applied === 'tool'
      ? '已切换为工具调用模式，新请求将把 JSON Schema 作为函数参数提交。'
      : '已切换为 JSON 模式，新请求使用 response_format 而不提交工具。';
    if (feedback) { feedback.textContent = message; feedback.className = 'form-feedback success'; }
    showAlert('settings-alert', message, 'success');
    return applied;
  } catch (error) {
    const message = `切换失败：${error.message}`;
    if (feedback) { feedback.textContent = message; feedback.className = 'form-feedback error'; }
    showAlert('settings-alert', message, 'error');
    return null;
  } finally {
    setBusy(target, false);
  }
}

async function saveProviderPlan(event) {
  event.preventDefault();
  const button = $('#save-provider-plan');
  const feedback = $('#provider-plan-feedback');
  setBusy(button, true, '保存中…');
  feedback.textContent = '';
  feedback.className = 'form-feedback';
  try {
    const result = await apiFetch('/api/v1/settings/provider-plans', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({name: $('#provider-plan-name').value.trim()}),
    });
    $('#provider-plan-name').value = '';
    await loadSettings();
    feedback.textContent = `已保存方案“${result.plan.name}”，并标记为当前配置。`;
    feedback.className = 'form-feedback success';
  } catch (error) {
    feedback.textContent = `保存方案失败：${error.message}`;
    feedback.className = 'form-feedback error';
  } finally {
    setBusy(button, false);
  }
}

async function activateProviderPlan(planId, button) {
  const plan = state.providerPlans.find((item) => item.id === planId);
  if (!plan) return;
  const feedback = $('#provider-plan-feedback');
  setBusy(button, true, '切换中…');
  feedback.textContent = '';
  feedback.className = 'form-feedback';
  try {
    const result = await apiFetch(`/api/v1/settings/provider-plans/${encodeURIComponent(planId)}/activate`, {method: 'POST'});
    await loadSettings();
    const message = result.embedding_rebuild_started
      ? `已切换到“${plan.name}”。存量段落正在后台重新向量化，期间检索可能暂时退化为关键词召回。`
      : `已切换到“${plan.name}”，新请求将使用该方案。`;
    feedback.textContent = message;
    feedback.className = 'form-feedback success';
    showAlert('settings-alert', message, 'success');
  } catch (error) {
    feedback.textContent = `切换方案失败：${error.message}`;
    feedback.className = 'form-feedback error';
    showAlert('settings-alert', `切换方案失败：${error.message}`, 'error');
  } finally {
    setBusy(button, false);
  }
}

async function deleteProviderPlan(plan, button) {
  if (!window.confirm(`确定删除方案“${plan.name}”吗？当前实际运行配置不会改变。`)) return;
  setBusy(button, true, '删除中…');
  try {
    await apiFetch(`/api/v1/settings/provider-plans/${encodeURIComponent(plan.id)}`, {method: 'DELETE'});
    await loadSettings();
    $('#provider-plan-feedback').textContent = `已删除方案“${plan.name}”。`;
    $('#provider-plan-feedback').className = 'form-feedback success';
  } catch (error) {
    $('#provider-plan-feedback').textContent = `删除方案失败：${error.message}`;
    $('#provider-plan-feedback').className = 'form-feedback error';
  } finally {
    setBusy(button, false);
  }
}

async function resetProviderSettings(group) {
  if (!window.confirm(`确定恢复${group === 'ai' ? '主 AI' : 'Embedding'}的部署默认配置吗？`)) return;
  const button = $(`#reset-${group}-provider`);
  const feedback = $(`#${group}-provider-feedback`);
  setBusy(button, true, '正在恢复…');
  feedback.textContent = '';
  feedback.className = 'form-feedback';
  try {
    const result = await apiFetch(`/api/v1/settings/providers/${group}`, {method: 'DELETE'});
    await loadSettings();
    const message = group === 'embedding' && result.embedding_rebuild_started
      ? '已恢复部署默认值；存量段落正在后台重新向量化。'
      : '已恢复部署默认配置。';
    feedback.textContent = message;
    feedback.className = 'form-feedback success';
    showAlert('settings-alert', message, 'success');
  } catch (error) {
    feedback.textContent = `恢复失败：${error.message}`;
    feedback.className = 'form-feedback error';
    showAlert('settings-alert', `恢复部署默认配置失败：${error.message}`, 'error');
  } finally {
    setBusy(button, false);
  }
}

function switchPage(page) {
  $$('.nav-item').forEach((item) => item.classList.toggle('active', item.dataset.page === page));
  $$('.page').forEach((section) => section.classList.toggle('active', section.id === `page-${page}`));
  if (page === 'settings') loadSettings();
  if (page === 'ingestion') loadDocuments();
  if (page === 'code') loadCodeDocuments();
  if (page === 'glossary') loadGlossary();
  if (page === 'chat') {
    renderChatThread();
    void loadConversations();
  }
}

function bindEvents() {
  $$('.nav-item').forEach((item) => item.addEventListener('click', () => switchPage(item.dataset.page)));
  $('#knowledge-base-select')?.addEventListener('change', (event) => switchKnowledgeBase(event.target.value));
  $('#create-knowledge-base')?.addEventListener('click', createKnowledgeBase);
  $('#choose-file').addEventListener('click', () => $('#file-input').click());
  $('#file-input').addEventListener('change', (event) => { enqueueDocumentFiles(event.target.files); event.target.value = ''; });
  const drop = $('#drop-zone');
  ['dragenter', 'dragover'].forEach((type) => drop.addEventListener(type, (event) => { event.preventDefault(); drop.classList.add('dragover'); }));
  ['dragleave', 'drop'].forEach((type) => drop.addEventListener(type, (event) => { event.preventDefault(); drop.classList.remove('dragover'); }));
  drop.addEventListener('drop', (event) => enqueueDocumentFiles(event.dataTransfer.files));
  $('#upload-queue').addEventListener('click', handleUploadQueueAction);
  drop.addEventListener('keydown', (event) => { if (event.key === 'Enter' || event.key === ' ') $('#file-input').click(); });
  $('#refresh-documents').addEventListener('click', loadDocuments);
  $('#documents-prev').addEventListener('click', () => { state.documentsPage.offset = Math.max(0, state.documentsPage.offset - state.documentsPage.limit); loadDocuments(); });
  $('#documents-next').addEventListener('click', () => { state.documentsPage.offset += state.documentsPage.limit; loadDocuments(); });
  $('#choose-code-folder').addEventListener('click', () => $('#code-folder-input').click());
  $('#code-folder-input').addEventListener('change', (event) => uploadCodeFolder(event.target.files));
  $('#choose-code-zip').addEventListener('click', () => $('#code-zip-input').click());
  $('#code-zip-input').addEventListener('change', (event) => uploadCodeZip(event.target.files[0]));
  $('#refresh-code').addEventListener('click', loadCodeDocuments);
  $('#code-prev').addEventListener('click', () => { state.codePage.offset = Math.max(0, state.codePage.offset - state.codePage.limit); loadCodeDocuments(); });
  $('#code-next').addEventListener('click', () => { state.codePage.offset += state.codePage.limit; loadCodeDocuments(); });
  const codeDrop = $('#code-drop-zone');
  ['dragenter', 'dragover'].forEach((type) => codeDrop.addEventListener(type, (event) => { event.preventDefault(); codeDrop.classList.add('dragover'); }));
  ['dragleave', 'drop'].forEach((type) => codeDrop.addEventListener(type, (event) => { event.preventDefault(); codeDrop.classList.remove('dragover'); }));
  codeDrop.addEventListener('drop', (event) => {
    const file = event.dataTransfer.files[0];
    if (file && /\.zip$/i.test(file.name)) uploadCodeZip(file);
    else showAlert('code-alert', '请拖入 .zip，或点「选择代码文件夹」上传整个目录。', 'info');
  });
  codeDrop.addEventListener('keydown', (event) => { if (event.key === 'Enter' || event.key === ' ') $('#code-folder-input').click(); });
  $('#search-button').addEventListener('click', () => runSearch(false));
  $('#answer-button').addEventListener('click', () => runSearch(true));
  $('#divergent-button').addEventListener('click', () => runSearch(true, 'divergent'));
  $('#question-input').addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && event.shiftKey) {
      event.preventDefault();
      $('#answer-button').click();
    }
  });
  $('#new-conversation').addEventListener('click', () => { if (!state.chat.sending) startNewChat(); });
  $('#rename-conversation').addEventListener('click', () => {
    if (state.chat.activeId) renameChatConversation(state.chat.activeId);
  });
  $('#delete-conversation').addEventListener('click', () => {
    if (state.chat.activeId) deleteChatConversation(state.chat.activeId);
  });
  $('#chat-form').addEventListener('submit', (event) => { event.preventDefault(); sendChatMessage(); });
  $('#chat-input').addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      sendChatMessage();
    }
  });
  $('#conversation-list').addEventListener('click', (event) => {
    const item = event.target.closest('[data-conversation-id]');
    if (!item) return;
    const action = event.target.closest('[data-conversation-action]');
    const conversationId = item.dataset.conversationId;
    if (action?.dataset.conversationAction === 'rename') {
      event.stopPropagation();
      renameChatConversation(conversationId);
      return;
    }
    if (action?.dataset.conversationAction === 'delete') {
      event.stopPropagation();
      deleteChatConversation(conversationId);
      return;
    }
    openChatConversation(conversationId);
  });
  $('#conversation-list').addEventListener('keydown', (event) => {
    if (event.key !== 'Enter' && event.key !== ' ') return;
    const item = event.target.closest('[data-conversation-id]');
    if (!item || event.target.closest('button')) return;
    event.preventDefault();
    openChatConversation(item.dataset.conversationId);
  });
  $('#chat-thread').addEventListener('click', (event) => {
    const feedbackButton = event.target.closest('.answer-feedback button');
    if (feedbackButton) { handleFeedback(feedbackButton); return; }
    const followup = event.target.closest('.followup-item');
    if (followup) {
      const input = $('#chat-input');
      input.value = followup.dataset.followup || '';
      input.focus();
      return;
    }
    const readerButton = event.target.closest('.reader-open');
    if (readerButton) { event.preventDefault(); openReader(readerButton); return; }
    const link = event.target.closest('.answer-citation a');
    if (!link) return;
    const target = document.getElementById(link.getAttribute('href').slice(1));
    if (!target) return;
    // 证据卡片嵌在“证据 N 条”的 details 里，必须把祖先 details 一并展开。
    for (let node = target; node; node = node.parentElement) {
      if (node.tagName === 'DETAILS') node.open = true;
    }
    target.setAttribute('tabindex', '-1');
    target.focus({preventScroll: true});
    target.scrollIntoView({block: 'nearest'});
  });
  $('#refresh-settings').addEventListener('click', loadSettings);
  $('#ai-provider-form').addEventListener('submit', (event) => { event.preventDefault(); saveProviderSettings('ai'); });
  $('#save-structured-output').addEventListener('click', () => saveStructuredOutput($('#ai-structured-mode').value));
  document.addEventListener('click', async (event) => {
    const button = event.target.closest('[data-switch-structured]');
    if (!button) return;
    const applied = await saveStructuredOutput(button.dataset.switchStructured, button);
    const feedback = button.parentElement?.querySelector('.form-feedback');
    if (applied && feedback) {
      feedback.textContent = `已切换为${applied === 'tool' ? '工具调用' : 'JSON 模式'}，请重新提问。`;
      feedback.className = 'form-feedback success';
    }
  });
  $('#embedding-provider-form').addEventListener('submit', (event) => { event.preventDefault(); saveProviderSettings('embedding'); });
  $('#provider-plan-form').addEventListener('submit', saveProviderPlan);
  $('#reset-ai-provider').addEventListener('click', () => resetProviderSettings('ai'));
  $('#reset-embedding-provider').addEventListener('click', () => resetProviderSettings('embedding'));
  $('#refresh-glossary').addEventListener('click', loadGlossary);
  $('#glossary-prev').addEventListener('click', () => { state.glossary.offset = Math.max(0, state.glossary.offset - state.glossary.limit); loadGlossary(); });
  $('#glossary-next').addEventListener('click', () => { state.glossary.offset += state.glossary.limit; loadGlossary(); });
  $('#glossary-body').addEventListener('click', (event) => {
    const button = event.target.closest('.remove-glossary-entry');
    if (button) removeGlossaryEntry(button);
  });
  $('#glossary-entries').addEventListener('click', (event) => {
    const button = event.target.closest('.remove-glossary-entry');
    if (button) removeGlossaryEntry(button);
  });
  $('#glossary-modal').addEventListener('click', (event) => { if (event.target.closest('[data-glossary-close]')) setGlossaryOpen(false); });
  document.addEventListener('keydown', (event) => { if (event.key === 'Escape' && !$('#glossary-modal').classList.contains('hidden')) setGlossaryOpen(false); });
  $('#glossary-choice-modal').addEventListener('click', (event) => {
    const strategyButton = event.target.closest('[data-glossary-strategy]');
    if (strategyButton && !state.glossaryChoiceBusy && state.glossaryChoiceButton) {
      buildGlossary(state.glossaryChoiceButton, strategyButton.dataset.glossaryStrategy);
      return;
    }
    if (event.target.closest('[data-glossary-choice-cancel]')) closeGlossaryChoice();
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && !$('#glossary-choice-modal').classList.contains('hidden')) closeGlossaryChoice();
  });
  $('#text-input-modal').addEventListener('click', (event) => {
    if (event.target.closest('[data-text-input-cancel]')) closeTextInput();
  });
  $('#text-input-form').addEventListener('submit', (event) => {
    event.preventDefault();
    const value = $('#text-input-value').value.trim();
    if (value) closeTextInput(value);
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && !$('#text-input-modal').classList.contains('hidden')) closeTextInput();
  });
  $('#copy-diagnostics').addEventListener('click', copyDiagnostics);
  $('#clear-knowledge-base').addEventListener('click', clearKnowledgeBase);
  const iterativeBox = $('#iterative-retrieval-input');
  if (iterativeBox) {
    iterativeBox.checked = window.localStorage.getItem('kb-iterative-retrieval') !== 'off';
    iterativeBox.addEventListener('change', () => window.localStorage.setItem('kb-iterative-retrieval', iterativeBox.checked ? 'on' : 'off'));
  }
  const debugBox = $('#debug-toggle');
  if (debugBox) {
    applyDebugMode(window.localStorage.getItem('kb-debug') === 'on');
    debugBox.checked = state.debug;
    debugBox.addEventListener('change', () => {
      applyDebugMode(debugBox.checked);
      window.localStorage.setItem('kb-debug', debugBox.checked ? 'on' : 'off');
    });
  }
  const sourceDocBox = $('#source-document-input');
  const sourceCodeBox = $('#source-code-input');
  if (sourceDocBox && sourceCodeBox) {
    const stored = window.localStorage.getItem('kb-source-kinds');
    if (stored) {
      sourceDocBox.checked = stored.split(',').includes('document');
      sourceCodeBox.checked = stored.split(',').includes('code');
    }
    const persistSourceKinds = () => {
      const kinds = selectedSourceKinds();
      if (kinds.length) window.localStorage.setItem('kb-source-kinds', kinds.join(','));
    };
    sourceDocBox.addEventListener('change', persistSourceKinds);
    sourceCodeBox.addEventListener('change', persistSourceKinds);
  }
  const apiInput = $('#api-base-input');
  if (apiInput) {
    apiInput.value = apiBase;
    apiInput.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') {
        event.preventDefault();
        switchBackend(apiInput.value);
      }
    });
  }
  const apiApply = $('#api-base-apply');
  if (apiApply) apiApply.addEventListener('click', () => switchBackend($('#api-base-input')?.value || ''));
  const apiReset = $('#api-base-reset');
  if (apiReset) apiReset.addEventListener('click', () => switchBackend(''));
}

async function initialize() {
  bindEvents();
  await loadKnowledgeBases();
  await Promise.all([loadSettings(), loadDocuments(), loadGlossary()]);
  setInterval(() => { if (state.documents.some((doc) => doc.status === 'processing')) loadDocuments(); }, 5000);
  setInterval(() => { if (state.codeDocuments.some((doc) => doc.status === 'processing')) loadCodeDocuments(); }, 5000);
}

initialize();
