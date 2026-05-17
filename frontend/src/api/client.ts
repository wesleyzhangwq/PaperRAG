import axios from 'axios'

export const http = axios.create({
  baseURL: '/api',
  timeout: 120_000,
})

export interface Source {
  paper_id: string
  title: string
  authors: string[]
  year?: number | null
  primary_category?: string | null
  doi?: string | null
  arxiv_url?: string | null
  score?: number | null
  page_num?: number | null
  chunk_index?: number | null
  snippet?: string | null
}

export interface ChatResponse {
  answer: string
  sources: Source[]
  used_chunks: number
}

export interface ChatFilter {
  category?: string | null
  year_min?: number | null
  paper_ids?: string[] | null
}

export interface PaperSummary {
  paper_id: string
  title: string
  authors: string[]
  year: number
  primary_category: string
  categories: string[]
  doi?: string | null
  abstract?: string | null
  arxiv_url?: string | null
  ingest_status: string
  num_chunks: number
}

export async function chat(query: string, filter?: ChatFilter, session_id = 'default'): Promise<ChatResponse> {
  const r = await http.post<ChatResponse>('/chat', { query, filter, session_id })
  return r.data
}

export async function listPapers(params: {
  category?: string
  year_min?: number
  q?: string
  limit?: number
  offset?: number
}): Promise<{ total: number; items: PaperSummary[] }> {
  const r = await http.get('/papers', { params })
  return r.data
}

export async function uploadPdf(file: File, title?: string): Promise<{ paper_id: string; status: string; num_chunks: number; message?: string }> {
  const fd = new FormData()
  fd.append('file', file)
  if (title) fd.append('title', title)
  const r = await http.post('/upload', fd, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return r.data
}

export async function health() {
  const r = await http.get('/health')
  return r.data
}

export async function chatStream(
  query: string,
  filter?: ChatFilter,
  session_id = 'default',
  onToken?: (token: string) => void,
  onSources?: (sources: Source[]) => void,
  onDone?: () => void,
  onError?: (err: Error) => void,
): Promise<void> {
  const resp = await fetch('/api/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, filter, session_id }),
  })

  if (!resp.ok || !resp.body) {
    onError?.(new Error(`HTTP ${resp.status}`))
    return
  }

  const reader = resp.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''

    let currentEvent = ''
    for (const line of lines) {
      if (line.startsWith('event: ')) {
        currentEvent = line.slice(7)
      } else if (line.startsWith('data: ')) {
        const data = JSON.parse(line.slice(6))
        if (currentEvent === 'token') onToken?.(data.t)
        else if (currentEvent === 'sources') onSources?.(data.sources)
        else if (currentEvent === 'done') onDone?.()
        currentEvent = ''
      }
    }
  }
}
