// 富文本渲染：Markdown + KaTeX 公式 + <ref id="N" /> 引用上标
//
// 数据契约：
//  - LLM 回答为 Markdown 文本，行内公式 $...$、独立行公式 $$...$$；
//  - 引用标签格式 <ref id="N" />（与后端 rag/citation.py 的 CITATION_PATTERN 对齐，
//    此处额外容忍漏写自闭合斜杠的变体）。
import MarkdownIt from 'markdown-it'
import { katex } from '@mdit/plugin-katex'
import DOMPurify from 'dompurify'

/** 引用标签：<ref id="2" />（容错 <ref id="2">）。 */
const REF_RE = /<ref\s+id="(\d+)"\s*\/?>/g

/** 块级公式 $$...$$（跨行允许）。 */
const DISPLAY_RE = /\$\$([\s\S]+?)\$\$/g

/** 占位符前缀/后缀（普通文本几乎不可能撞到，还原前不受 markdown 影响）。 */
const PLACEHOLDER_RE = /@@CIT:(\d+)@@/g

const md = new MarkdownIt({
  html: false, // 原生 HTML 一律转义（引用标签已在渲染前替换为占位符）
  linkify: true,
  breaks: true, // 单换行 → <br>，保真流式回答的行结构
}).use(katex, { throwOnError: false })

/**
 * markdown-it 的 $$...$$ 块级规则只识别「独占一行」的公式；LLM 有时会把
 * $$...$$ 嵌在段落行内（如“……，有$$a<b$$则……”）。这里把这类行内 $$ 块
 * 用空行隔开，使其退化为独立块级公式再交给 katex 插件渲染。
 */
function isolateDisplayMath(src: string): string {
  return src.replace(DISPLAY_RE, (_m, tex: string) => `\n\n$$${tex}$$\n\n`)
}

export interface RichCitationEntry {
  kp_path?: string
  page_ref?: string
  uuid?: string
}

export interface RichRenderResult {
  html: string
  /** 正文中实际出现过的引用 id（按首次出现顺序，去重） */
  refIds: number[]
}

/**
 * 渲染问答富文本。
 * ref 标签 → 占位符 → markdown-it(+katex) → 还原为上标 → DOMPurify 净化。
 */
export function renderRich(content: string): RichRenderResult {
  const refIds: number[] = []
  const seen = new Set<number>()

  const withRefPlaceholders = content.replace(REF_RE, (_m, idStr: string) => {
    const id = Number(idStr)
    if (!seen.has(id)) {
      seen.add(id)
      refIds.push(id)
    }
    return `@@CIT:${idStr}@@`
  })
  const isolated = isolateDisplayMath(withRefPlaceholders)

  const rawHtml = md.render(isolated)
  const restored = rawHtml.replace(PLACEHOLDER_RE, (_m, idStr: string) => {
    const id = Number(idStr)
    return `<sup class="citation-ref" data-ref="${id}">[${id}]</sup>`
  })

  return { html: DOMPurify.sanitize(restored), refIds }
}
