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
  year_max?: number | null
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
  year_max?: number
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
