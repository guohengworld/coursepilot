<script setup lang="ts">
// 富文本回答渲染器：Markdown + LaTeX 公式排版 + 可点击引用上标与来源面板。
// 渲染走 utils/markdown.ts 的 renderRich（净化为安全 HTML），
// 输入做 rAF 级去抖，流式逐 token 追加时不会每 token 全量重渲。
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import { renderRich, type RichCitationEntry } from '@/utils/markdown'

const props = withDefaults(
  defineProps<{
    content: string
    /** ref id → 来源信息（kp_path / page_ref 等），缺省时仅展示上标数字 */
    citations?: Record<string, RichCitationEntry>
    /** 隐藏底部来源面板（如纯文本错误/状态气泡） */
    hideRefs?: boolean
  }>(),
  { citations: () => ({}), hideRefs: false },
)

const bodyRef = ref<HTMLElement | null>(null)
const panelRef = ref<HTMLElement | null>(null)
const renderedHtml = ref('')
const refIds = ref<number[]>([])
const activeRef = ref<number | null>(null)

const showPanel = computed(() => !props.hideRefs && refIds.value.length > 0)
const panelEntries = computed(() =>
  refIds.value.map((id) => ({ id, entry: props.citations?.[String(id)] })),
)

function doRender(content: string) {
  const res = renderRich(content)
  renderedHtml.value = res.html
  refIds.value = res.refIds
}

// rAF 级去抖：同一帧内多次 content 变化只渲染一次最新值
let rafId: number | null = null
function scheduleRender() {
  if (rafId !== null) return
  rafId = requestAnimationFrame(() => {
    rafId = null
    doRender(props.content)
  })
}
watch(() => props.content, scheduleRender, { immediate: true })

function scrollIntoViewIfInside(el: HTMLElement | null) {
  if (el) el.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
}

function onBodyClick(e: MouseEvent) {
  const target = e.target as HTMLElement
  const sup = target.closest('.citation-ref') as HTMLElement | null
  if (sup) {
    const id = Number(sup.dataset.ref)
    activeRef.value = id
    scrollIntoViewIfInside(
      panelRef.value?.querySelector(`.refs-item[data-ref="${id}"]`) as HTMLElement | null,
    )
    return
  }
  // 点击正文其他位置取消高亮
  if (activeRef.value !== null) activeRef.value = null
}

function onPanelItemClick(id: number) {
  activeRef.value = id
  // 反向滚到正文中该引用的第一个上标
  scrollIntoViewIfInside(
    bodyRef.value?.querySelector(`.citation-ref[data-ref="${id}"]`) as HTMLElement | null,
  )
}

onBeforeUnmount(() => {
  if (rafId !== null) cancelAnimationFrame(rafId)
})
</script>

<template>
  <div class="rich-text-root">
    <div ref="bodyRef" class="rich-text" @click="onBodyClick">
      <!-- 空内容时回退为普通文本；净化为空的非空内容也安全（v-html 防注入） -->
      <span v-if="!renderedHtml" class="rich-empty">{{ content }}</span>
      <span v-else v-html="renderedHtml"></span>
    </div>

    <!-- 参考来源面板 -->
    <div v-if="showPanel" ref="panelRef" class="refs-panel">
      <div class="refs-title">参考来源</div>
      <div
        v-for="e in panelEntries"
        :key="e.id"
        :data-ref="e.id"
        :class="['refs-item', { active: activeRef === e.id }]"
        @click="onPanelItemClick(e.id)"
      >
        <span class="refs-num">[{{ e.id }}]</span>
        <span class="refs-meta">
          <span v-if="e.entry?.kp_path" class="refs-kp" :title="e.entry.kp_path">{{ e.entry.kp_path }}</span>
          <span v-if="e.entry?.page_ref" class="refs-page">（{{ e.entry.page_ref }}）</span>
          <span v-if="!e.entry?.kp_path && !e.entry?.page_ref" class="refs-fallback">暂无来源信息</span>
        </span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.rich-text {
  line-height: 1.7;
  font-size: 14px;
  word-break: break-word;
  color: var(--el-text-color-primary);
}
.rich-text :deep(p) {
  margin: 0.35em 0;
}
.rich-text :deep(h1),
.rich-text :deep(h2),
.rich-text :deep(h3),
.rich-text :deep(h4) {
  margin: 0.7em 0 0.4em;
  font-weight: 600;
  line-height: 1.4;
}
.rich-text :deep(h1) { font-size: 1.5em; }
.rich-text :deep(h2) { font-size: 1.3em; }
.rich-text :deep(h3) { font-size: 1.15em; }
.rich-text :deep(h4) { font-size: 1.05em; }
.rich-text :deep(ul),
.rich-text :deep(ol) {
  padding-left: 1.6em;
  margin: 0.35em 0;
}
.rich-text :deep(li) {
  margin: 0.15em 0;
}
.rich-text :deep(blockquote) {
  margin: 0.4em 0;
  padding: 0.1em 1em;
  border-left: 3px solid var(--el-border-color-darker);
  color: var(--el-text-color-secondary);
  background: var(--el-fill-color-lighter);
  border-radius: 0 4px 4px 0;
}
.rich-text :deep(code) {
  font-family: var(--el-font-family-mono, 'Consolas', monospace);
  background: var(--el-fill-color-light);
  padding: 0.1em 0.35em;
  border-radius: 4px;
  font-size: 0.92em;
}
.rich-text :deep(pre) {
  background: var(--el-fill-color-light);
  padding: 0.6em 0.9em;
  border-radius: 6px;
  overflow-x: auto;
  margin: 0.5em 0;
}
.rich-text :deep(pre code) {
  background: transparent;
  padding: 0;
}
.rich-text :deep(table) {
  border-collapse: collapse;
  margin: 0.5em 0;
}
.rich-text :deep(th),
.rich-text :deep(td) {
  border: 1px solid var(--el-border-color-light);
  padding: 4px 10px;
  text-align: left;
}
.rich-text :deep(img) {
  max-width: 100%;
}
.rich-text :deep(a) {
  color: var(--el-color-primary);
  text-decoration: none;
}
.rich-text :deep(.katex-display) {
  overflow-x: auto;
  overflow-y: hidden;
  padding: 2px 0;
  margin: 0.4em 0;
}
.rich-text :deep(.katex) {
  font-size: 1.05em;
}
/* 引用上标 [N] */
.rich-text :deep(.citation-ref) {
  color: var(--el-color-primary);
  cursor: pointer;
  user-select: none;
  font-size: 0.72em;
  font-weight: 600;
  margin: 0 1px;
}
.rich-text :deep(.citation-ref:hover) {
  text-decoration: underline;
  background: var(--el-color-primary-light-9);
  border-radius: 3px;
}
.rich-empty {
  white-space: pre-wrap;
}

/* 参考来源面板 */
.refs-panel {
  margin-top: 10px;
  border-top: 1px dashed var(--el-border-color-light);
  padding-top: 8px;
}
.refs-title {
  font-size: 12px;
  font-weight: 600;
  color: var(--el-text-color-secondary);
  margin-bottom: 6px;
}
.refs-item {
  display: flex;
  gap: 8px;
  align-items: baseline;
  padding: 4px 8px;
  border-radius: 5px;
  cursor: pointer;
  transition: background 0.15s;
}
.refs-item:hover {
  background: var(--el-fill-color-light);
}
.refs-item.active {
  background: var(--el-color-primary-light-9);
  box-shadow: inset 3px 0 0 var(--el-color-primary);
}
.refs-num {
  color: var(--el-color-primary);
  font-weight: 600;
  font-size: 12px;
  flex-shrink: 0;
}
.refs-meta {
  font-size: 12px;
  line-height: 1.6;
  min-width: 0;
}
.refs-kp {
  color: var(--el-text-color-regular);
}
.refs-page {
  color: var(--el-text-color-secondary);
}
.refs-fallback {
  color: var(--el-text-color-placeholder);
  font-style: italic;
}
</style>
