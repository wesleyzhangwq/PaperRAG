export interface StepTrace {
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

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
  thinking?: ThinkingStep[]
  timestamp: number
}

export interface ThinkingStep {
  index: number
  action: string
  reason: string
  status: 'pending' | 'running' | 'done' | 'failed'
  outputSummary?: string
  durationMs?: number
}
