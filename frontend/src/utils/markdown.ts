import MarkdownIt from 'markdown-it'
import type { Source } from '../types'

const md = new MarkdownIt({ html: false, linkify: true, breaks: true })

const CITATION_RE = /\[arxiv:(\d{4}\.\d{4,6})\]/g

/**
 * Render markdown with citation pills.
 * Citations are rendered as spans with data-paper-id for the Vue component to bind popovers.
 */
export function renderMarkdown(text: string, sources?: Source[]): string {
  if (!text) return ''

  const citedIds: string[] = []
  const processed = text.replace(CITATION_RE, (_, id: string) => {
    if (!citedIds.includes(id)) citedIds.push(id)
    const idx = citedIds.indexOf(id) + 1
    return `<span class="citation-pill inline-flex items-center px-1.5 py-0.5 rounded text-xs font-medium bg-amber-100 text-amber-700 cursor-pointer hover:bg-amber-200 transition" data-paper-id="${id}" data-citation-index="${idx}">[${idx}]</span>`
  })

  return md.render(processed)
}

/**
 * Extract cited paper IDs in order from the answer text.
 */
export function extractCitedIds(text: string): string[] {
  const ids: string[] = []
  let match
  const re = /\[arxiv:(\d{4}\.\d{4,6})\]/g
  while ((match = re.exec(text)) !== null) {
    if (!ids.includes(match[1])) ids.push(match[1])
  }
  return ids
}
