import MarkdownIt from 'markdown-it'
import type { Source } from '../types'

const md = new MarkdownIt({ html: false, linkify: true, breaks: true })

const CITATION_RE = /\[(?:arxiv|source):([A-Za-z0-9][A-Za-z0-9._:-]{0,63})\]/g
const CITATION_TOKEN_PREFIX = 'PAPERRAG_CITATION_'
const THINK_BLOCK_RE = /<think>[\s\S]*?<\/think>/gi
const THINK_OPEN_RE = /<think>[\s\S]*$/i
const ESCAPED_THINK_BLOCK_RE = /&lt;think&gt;[\s\S]*?&lt;\/think&gt;/gi
const ESCAPED_THINK_OPEN_RE = /&lt;think&gt;[\s\S]*$/i

/**
 * Render markdown with citation pills.
 * Citations are rendered as spans with data-paper-id for the Vue component to bind popovers.
 */
export function renderMarkdown(text: string, sources?: Source[]): string {
  if (!text) return ''

  const citedIds: string[] = []
  const tokens: Record<string, string> = {}
  const visibleText = stripHiddenReasoning(text)
  const processed = visibleText.replace(CITATION_RE, (_, id: string) => {
    if (!citedIds.includes(id)) citedIds.push(id)
    const idx = citedIds.indexOf(id) + 1
    const token = `${CITATION_TOKEN_PREFIX}${idx}__`
    tokens[token] = citationPill(id, idx, sources)
    return token
  })

  let html = md.render(processed)
  for (const [token, pill] of Object.entries(tokens)) {
    html = html.split(token).join(pill)
  }
  return html
}

function stripHiddenReasoning(text: string): string {
  return text
    .replace(THINK_BLOCK_RE, '')
    .replace(THINK_OPEN_RE, '')
    .replace(ESCAPED_THINK_BLOCK_RE, '')
    .replace(ESCAPED_THINK_OPEN_RE, '')
    .replace(/^\n+/, '')
    .trimEnd()
}

/**
 * Extract cited paper IDs in order from the answer text.
 */
export function extractCitedIds(text: string): string[] {
  const ids: string[] = []
  let match
  const re = /\[(?:arxiv|source):([A-Za-z0-9][A-Za-z0-9._:-]{0,63})\]/g
  while ((match = re.exec(text)) !== null) {
    if (!ids.includes(match[1])) ids.push(match[1])
  }
  return ids
}

function citationPill(id: string, idx: number, sources?: Source[]): string {
  const known = !!sources?.some(s => s.paper_id === id)
  const title = known ? '点击打开引用论文' : '引用论文未在来源列表中找到'
  return `<span class="citation-pill" data-paper-id="${id}" data-citation-index="${idx}" role="button" tabindex="0" title="${title}">[${idx}]</span>`
}
