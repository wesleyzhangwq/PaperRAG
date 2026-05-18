import MarkdownIt from 'markdown-it'
import type { Source } from '../types'

const md = new MarkdownIt({ html: false, linkify: true, breaks: true })

const CITATION_RE = /\[arxiv:(\d{4}\.\d{4,6})\]/g

export function renderMarkdown(text: string, sources?: Source[]): string {
  if (!text) return ''

  const citedIds: string[] = []
  const processed = text.replace(CITATION_RE, (_, id: string) => {
    if (!citedIds.includes(id)) citedIds.push(id)
    const idx = citedIds.indexOf(id) + 1
    const source = sources?.find(s => s.paper_id === id)
    const title = source?.title || id
    return `<span class="citation-pill inline-flex items-center px-1.5 py-0.5 rounded text-xs font-medium bg-amber-100 text-amber-700 cursor-pointer hover:bg-amber-200 transition" title="${title}">[${idx}]</span>`
  })

  return md.render(processed)
}
