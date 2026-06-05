export interface StepTrace {
  index?: number
  node: string
  action: string
  input_summary: string
  output_summary: string
  duration_ms: number
}

export interface Source {
  paper_id: string
  title: string
  authors: string[]
  year: number | null
  primary_category?: string
  doi?: string
  arxiv_url: string
  score?: number
  page_num?: number
  snippet?: string
}

export interface SSEIntent {
  type: 'simple' | 'complex' | 'comparison'
  entities: string[]
  complexity: 'low' | 'medium' | 'high'
}

export interface SSEPlan {
  steps: { action: string; reason: string; params?: Record<string, unknown> }[]
  total_steps: number
}

export interface SSEReflection {
  passed: boolean
  citation_ok: boolean
  completeness_ok: boolean
  logic_ok: boolean
}

export interface ToolCallEvent {
  index: number
  action: string
  params: Record<string, unknown>
  reason: string
}

export interface ToolResultEvent {
  index: number
  action: string
  duration_ms: number
  summary: string
  detail: Record<string, unknown>
}

// ---------------------------------------------------------------------------
// Productized presentation payload (built by the backend `presentation_node`)
// ---------------------------------------------------------------------------
export type Confidence = 'high' | 'medium' | 'low'
export type Relevance = 'high' | 'medium' | 'low'
export type StepStatus = 'completed' | 'warning' | 'error' | 'running' | 'pending'

export interface SourceCard {
  paper_id: string
  title: string
  authors?: string[]
  year?: number | null
  primary_category?: string | null
  arxiv_url: string
  relevance: Relevance
  hit_count: number
  summary: string
  snippets: string[]
}

export interface RetrievalSummary {
  total_chunks: number
  total_papers: number
  cited_papers?: number
  web_results?: number
  main_topics: string[]
  is_fallback: boolean
  narrative: string
}

export interface PresentationStep {
  index: number
  name: string                    // Chinese user-facing label
  action: string                  // internal action (kept for debug)
  status: StepStatus
  user_message: string
  duration_ms: number
  debug: {
    tool: string
    params: Record<string, unknown>
    reason: string
    raw_summary: string
    extra: Record<string, unknown>
  }
}

export interface Presentation {
  answer: string
  confidence: Confidence
  confidence_reason: string
  sources: SourceCard[]
  retrieval_summary: RetrievalSummary
  steps: PresentationStep[]
}

// ---------------------------------------------------------------------------
// Live thinking step (in-flight before presentation arrives)
// ---------------------------------------------------------------------------
export interface ThinkingStep {
  index: number
  action: string
  reason: string
  status: 'pending' | 'running' | 'done' | 'failed'
  outputSummary?: string
  durationMs?: number
  detailParams?: Record<string, unknown>
  detailResult?: Record<string, unknown>
  detailSource?: 'presentation'
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  reasoning?: string
  sources?: Source[]
  thinking?: ThinkingStep[]
  toolCalls?: ToolCallEvent[]
  toolResults?: ToolResultEvent[]
  presentation?: Presentation | null
  elapsedMs?: number
  timestamp: number
  pending?: boolean
}

export interface Conversation {
  id: string
  title: string
  pinned: boolean
  created_at: string
  updated_at: string
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

export interface PaperListResponse {
  total: number
  items: PaperSummary[]
}

export interface CorpusRepresentativePaper {
  paper_id: string
  title: string
  year: number | null
  primary_category: string
  arxiv_url: string
}

export interface CorpusTopicBucket {
  key: string
  label: string
  description: string
  paper_count: number
  chunk_count: number
  representative_papers: CorpusRepresentativePaper[]
}

export interface CorpusOverviewResponse {
  total_papers: number
  total_chunks: number
  year_min: number | null
  year_max: number | null
  topic_buckets: CorpusTopicBucket[]
  suggested_questions: string[]
  generated_at: string
}

export interface UploadResponse {
  job_id?: string | null
  paper_id: string
  status: string
  num_chunks: number
  message?: string | null
}

export interface UploadJob {
  job_id: string
  paper_id: string
  filename: string
  title: string
  status: string
  num_chunks: number
  message?: string | null
  created_at: string
  updated_at: string
}

export interface UploadJobListResponse {
  total: number
  items: UploadJob[]
}

export interface ServerMessage {
  id: number
  role: 'user' | 'assistant'
  content: string
  sources: Source[]
  thinking: StepTrace[]
  presentation?: Presentation | null
  elapsed_ms?: number | null
  created_at: string
}

export type Theme = 'light' | 'dark'
