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

// ---------------------------------------------------------------------------
// v2 SSE protocol — stage events with stable ids
// ---------------------------------------------------------------------------
export interface SSEStage {
  /** Stable id: pipeline stage key ("intent", "evidence", …) or "step:N". */
  id: string
  /** Pipeline stage kind; executor tool steps use "retrieve_step". */
  stage: string
  status: 'start' | 'done' | 'warning' | 'failed' | 'skipped'
  title: string
  summary?: string
  detail?: Record<string, unknown>
  duration_ms?: number
}

export interface SSEPlanStep {
  id: string
  action: string
  title: string
  reason: string
}

export interface SSEPlan {
  revision: number
  steps: SSEPlanStep[]
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

export type AgentExecutionPath = 'fast_local' | 'full_agentic' | 'fast_escalated'

export interface ComplexityDecision {
  policy_version: string
  mode: string
  initial_path: AgentExecutionPath
  final_path: AgentExecutionPath
  confidence: string
  reason_codes: string[]
  vetoes: string[]
  features: Record<string, string | number | boolean>
  escalated: boolean
}

export interface Presentation {
  answer: string
  confidence: Confidence
  confidence_reason: string
  sources: SourceCard[]
  retrieval_summary: RetrievalSummary
  steps: PresentationStep[]
  execution_path?: AgentExecutionPath | null
  complexity_decision?: ComplexityDecision | null
}

// ---------------------------------------------------------------------------
// Agent activity timeline (live during streaming + rebuilt on history load)
// ---------------------------------------------------------------------------
export type TimelineStatus = 'pending' | 'running' | 'done' | 'warning' | 'failed' | 'skipped'

export interface TimelineItem {
  /** Stable id — stage key or "step:N". Events upsert by id. */
  id: string
  /** 'stage' = pipeline node; 'step' = executor tool step (nested visual). */
  kind: 'stage' | 'step'
  title: string
  status: TimelineStatus
  summary?: string
  reason?: string
  detail?: Record<string, unknown>
  durationMs?: number
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  reasoning?: string
  sources?: Source[]
  timeline?: TimelineItem[]
  presentation?: Presentation | null
  elapsedMs?: number
  timestamp: number
  pending?: boolean
  /** True once answer tokens have started streaming for this turn. */
  answering?: boolean
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
  topic_bucket_key: string
  topic_bucket_label: string
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

export interface ArxivImportBatchResponse {
  total: number
  items: UploadResponse[]
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
