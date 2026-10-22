"""清洗前后只读文本对比；滚动联动仅在当前组件内生效。"""
import hashlib
from html import escape
from string import Template

from streamlit.components.v1 import html as component_html

PREVIEW_LIMIT = 60000


def _comparison_html(task_id, before, after, before_label="清洗前（含按需OCR结果）", after_label="清洗后",
    right_html=None, preview_limit=PREVIEW_LIMIT):
    if preview_limit is not None:
        before, after = before[:preview_limit], after[:preview_limit]
    state_key = hashlib.sha256((str(task_id) + "\0" + before_label + "\0" + before + "\0" +
        after_label + "\0" + (right_html if right_html is not None else after)).encode("utf-8")).hexdigest()
    right_pane = ('<div id="after-content" class="chunk-list" tabindex="0" role="region" '
        'aria-labelledby="after-label">' + right_html + '</div>') if right_html is not None else (
        '<textarea id="after-content" readonly spellcheck="false">\n' + escape(after) + '</textarea>')
    return Template("""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<style>
  :root { color-scheme: light dark; --text: #31333f; --surface: #f0f2f6; --border: #d5d7de; --accent: #0068c9; }
  * { box-sizing: border-box; }
  body { margin: 0; color: var(--text); font: 14px/1.5 system-ui, sans-serif; }
  .toolbar { display: flex; align-items: center; flex-wrap: wrap; gap: 12px; min-height: 44px; margin-bottom: 8px; }
  .toolbar label { display: flex; align-items: center; gap: 8px; min-height: 44px; cursor: pointer; }
  #sync-scroll { width: 18px; height: 18px; margin: 0; accent-color: var(--accent); }
  #scroll-mode { font-size: 13px; }
  .panes { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
  .pane label { display: flex; align-items: flex-end; height: 42px; margin-bottom: 8px; }
  textarea, .chunk-list { display: block; width: 100%; height: 340px; padding: 14px; resize: none;
    border: 1px solid var(--border); border-radius: 8px; background: var(--surface); color: var(--text);
    font: 16px/1.6 system-ui, sans-serif; overflow: auto; overflow-wrap: anywhere; scrollbar-gutter: stable; }
  textarea:focus-visible, .chunk-list:focus-visible, #sync-scroll:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  .chunk-card, .parent-group { border: 1px solid var(--border); border-radius: 6px; padding: 12px; margin-bottom: 12px; }
  .chunk-title { margin: 0 0 8px; font-size: 14px; font-weight: 600; }
  .chunk-body { white-space: pre-wrap; overflow-wrap: anywhere; }
  .chunk-source { font-size: 12px; margin-bottom: 8px; }
  .parent-text { border-bottom: 1px solid var(--border); padding-bottom: 12px; margin-bottom: 12px; }
  .chunk-card:last-child, .parent-group:last-child { margin-bottom: 0; }
  @media (prefers-color-scheme: dark) {
    :root { --text: #fafafa; --surface: #262730; --border: #555866; --accent: #83c9ff; }
  }
  @media (max-width: 480px) {
    .panes { gap: 8px; }
    textarea, .chunk-list { padding: 8px; }
    .chunk-card, .parent-group { padding: 8px; }
  }
</style>
</head>
<body>
  <div class="toolbar">
    <label for="sync-scroll"><input id="sync-scroll" type="checkbox" checked>同步滚动</label>
    <span id="scroll-mode" role="status" aria-live="polite"></span>
  </div>
  <div class="panes">
    <div class="pane">
      <label for="before-content">$before_label</label>
      <textarea id="before-content" readonly spellcheck="false">
$before_content</textarea>
    </div>
    <div class="pane">
      <label id="after-label" for="after-content">$after_label</label>
      $right_pane
    </div>
  </div>
<script>
(() => {
  const before = document.getElementById('before-content');
  const after = document.getElementById('after-content');
  const sync = document.getElementById('sync-scroll');
  const mode = document.getElementById('scroll-mode');
  const storageKey = 'rag-upload-compare:$state_key';
  const expected = new WeakMap();
  let leader = before;
  let saveFrame = 0;

  const range = pane => Math.max(0, pane.scrollHeight - pane.clientHeight);
  const progress = pane => range(pane) ? Math.max(0, Math.min(1, pane.scrollTop / range(pane))) : 0;
  function move(pane, ratio) {
    const top = Math.max(0, Math.min(1, Number(ratio) || 0)) * range(pane);
    if (Math.abs(pane.scrollTop - top) > 0.5) {
      pane.scrollTop = top;
      expected.set(pane, pane.scrollTop);
    }
  }
  function align(source) {
    move(source === before ? after : before, progress(source));
  }
  function save() {
    try {
      sessionStorage.setItem(storageKey, JSON.stringify({sync: sync.checked,
        before: progress(before), after: progress(after), leader: leader === before ? 'before' : 'after'}));
    } catch (_) { /* 浏览器禁止保存状态时仍可正常滚动。 */ }
  }
  function scheduleSave() {
    if (!saveFrame) saveFrame = requestAnimationFrame(() => { saveFrame = 0; save(); });
  }
  function updateMode() {
    mode.textContent = sync.checked ? '已开启：左右联动' : '已关闭：可分别滚动';
  }
  for (const pane of [before, after]) {
    pane.addEventListener('scroll', () => {
      const pending = expected.get(pane);
      expected.delete(pane);
      // 忽略另一侧联动产生的滚动事件，避免两边反复触发。
      if (pending === undefined || Math.abs(pane.scrollTop - pending) > 0.5) {
        leader = pane;
        if (sync.checked) align(pane);
      }
      scheduleSave();
    }, {passive: true});
  }
  sync.addEventListener('change', () => {
    if (sync.checked) align(leader);
    updateMode();
    save();
  });
  window.addEventListener('resize', () => {
    if (sync.checked) align(leader);
    scheduleSave();
  });
  try {
    const saved = JSON.parse(sessionStorage.getItem(storageKey));
    if (saved && typeof saved.sync === 'boolean') {
      sync.checked = saved.sync;
      leader = saved.leader === 'after' ? after : before;
      move(before, saved.before);
      move(after, saved.after);
      if (sync.checked) align(leader);
    }
  } catch (_) { /* 首次进入或状态不可用时，默认同步滚动。 */ }
  updateMode();
})();
</script>
</body>
</html>""").substitute(state_key=state_key, before_content=escape(before),
        before_label=escape(before_label), after_label=escape(after_label), right_pane=right_pane)


def render_content_compare(task_id, before, after):
    component_html(_comparison_html(task_id, before, after), height=450, scrolling=False)


def _chunk_card(chunk, title):
    metadata = chunk.get("metadata", {})
    source = ('<div class="chunk-source">来源页码：' + escape(str(metadata["page"])) + '</div>') if (
        metadata.get("page") is not None) else ""
    return ('<article class="chunk-card"><h4 class="chunk-title">' + escape(title) + '</h4>' + source +
        '<div class="chunk-body">' + escape(chunk.get("content", "")) + '</div></article>')


def _chunks_html(chunks, strategy):
    if strategy != "parent_child":
        return "".join(_chunk_card(chunk, f"片段 {chunk.get('index', index)}")
            for index, chunk in enumerate(chunks, 1))
    groups = {}
    for index, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        if metadata.get("parent_id"):
            group = metadata["parent_id"]
        elif metadata.get("parent_index") is not None:
            group = ("parent", metadata["parent_index"])
        else:
            group = ("missing", index)
        groups.setdefault(group, []).append(chunk)
    markup = []
    for index, children in enumerate(groups.values(), 1):
        metadata = children[0].get("metadata", {})
        parent_index = metadata.get("parent_index", index)
        content = metadata.get("parent_content")
        parent_text = content if content is not None else "本次历史分块未保留父块原文，请重新生成分块。"
        markup.append('<section class="parent-group"><h3 class="chunk-title">父块 ' +
            escape(str(parent_index)) + f' · {len(children)} 个子块</h3><div class="chunk-body parent-text">' +
            escape(parent_text) + '</div>' + "".join(_chunk_card(child,
                f"子块 {parent_index}.{child.get('metadata', {}).get('child_index', number)}")
                for number, child in enumerate(children, 1)) + '</section>')
    return "".join(markup)


def render_chunk_compare(task_id, original, chunks, strategy):
    component_html(_comparison_html("chunks:" + task_id, original, "", "原文（清洗后）", "分块内容",
        right_html=_chunks_html(chunks, strategy), preview_limit=None), height=450, scrolling=False)
