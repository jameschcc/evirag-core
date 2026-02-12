/* =============================================================================
 * EviRAG — Enterprise-grade High-performance Offline Knowledge Base with
 * Traceable Retrieval and Generation Architecture
 * Personal Research Project · Author: S.Chang (常帅)
 * Paper: 常帅. 企业级高性能离网知识库可溯源检索与生成架构[J].
 *        China Information World, 2026(2).
 *        DOI: 10.3969/j.issn.1671-3370.2026.02.032
 *        https://d.wanfangdata.com.cn/periodical/zgxxj202602032
 * ========================================================================== */

/*
 * Answer rendering wrapper around the vendored markdown-it build
 * (web/markdown-it.min.js, MIT, 14.1.0). The parser runs with html:false so
 * model output and evidence text can never inject markup; raw HTML shows up as
 * literal text instead. We only add the three behaviours this product needs:
 *   - [E1] / [E1, E2] citations rendered through the caller's callback
 *   - external links open in a new tab with rel=noopener
 *   - wide tables stay horizontally scrollable inside the answer card
 */
(() => {
  const escapeHtml = (value) => String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');

  // Minimal fallback for a failed vendor load: keep text readable, never HTML.
  function renderPlain(markdown) {
    const text = String(markdown || '').replace(/\r\n?/g, '\n').trim();
    if (!text) return '';
    return text.split(/\n{2,}/).map((block) => `<p>${escapeHtml(block).replace(/\n/g, '<br>')}</p>`).join('');
  }

  let parser = null;

  function createParser() {
    if (parser) return parser;
    parser = window.markdownit({
      html: false,
      linkify: true,
      breaks: true,
      typographer: false,
    });

    const defaultLinkOpen = parser.renderer.rules.link_open
      || ((tokens, idx, options, env, self) => self.renderToken(tokens, idx, options));
    parser.renderer.rules.link_open = (tokens, idx, options, env, self) => {
      const href = tokens[idx].attrGet('href') || '';
      if (/^https?:/i.test(href)) {
        tokens[idx].attrSet('target', '_blank');
        tokens[idx].attrSet('rel', 'noopener noreferrer');
      }
      return defaultLinkOpen(tokens, idx, options, env, self);
    };

    // Markdown images would issue third-party requests from model or evidence
    // text; keep only the alt text.
    parser.renderer.rules.image = (tokens, idx) => escapeHtml(tokens[idx].content || '');

    // GFM table output gets a scroll container so wide evidence tables do not
    // push the answer card sideways.
    parser.renderer.rules.table_open = () => '<div class="answer-table">\n<table>\n';
    parser.renderer.rules.table_close = () => '</table>\n</div>\n';

    // [E1] / [E1, E2]: the only custom syntax on top of CommonMark + GFM.
    // Inserted before "link" so code spans have already been tokenized.
    parser.inline.ruler.before('link', 'kb_citation', (state, silent) => {
      const src = state.src;
      const start = state.pos;
      if (src.charCodeAt(start) !== 0x5B) return false;                 // [
      if (start > 0 && src.charCodeAt(start - 1) === 0x21) return false; // ![alt]
      const match = src.slice(start).match(/^\[(E\d+(?:\s*,\s*E\d+)*)\]/);
      if (!match) return false;
      const end = start + match[0].length;
      if (src.charCodeAt(end) === 0x28) return false;                    // [E1](url) is a normal link
      if (!silent) {
        const token = state.push('kb_citation', '', 0);
        token.meta = {ids: match[1].split(/\s*,\s*/)};
      }
      state.pos = end;
      return true;
    });

    parser.renderer.rules.kb_citation = (tokens, idx, options, env) => {
      const renderCitation = env.renderCitation || ((id) => `[${escapeHtml(id)}]`);
      const ids = tokens[idx].meta?.ids || [];
      return ids.map((id) => renderCitation(id)).join('');
    };

    return parser;
  }

  function render(markdown, options = {}) {
    const text = String(markdown || '');
    if (!text.trim()) return '';
    if (typeof window.markdownit !== 'function') return renderPlain(text);
    const renderCitation = options.renderCitation || ((id) => `[${escapeHtml(id)}]`);
    return createParser().render(text, {renderCitation});
  }

  window.AnswerMarkdown = {render};
})();
