import api from './client'
import type { CorpusOverviewResponse, PaperListResponse } from '../types'

export async function listPapers(params: {
  q?: string
  category?: string
  year_min?: number
  year_max?: number
  limit?: number
} = {}): Promise<PaperListResponse> {
  const { data } = await api.get<PaperListResponse>('/papers', { params })
  return data
}

export async function getCorpusOverview(): Promise<CorpusOverviewResponse> {
  const { data } = await api.get<CorpusOverviewResponse>('/papers/overview')
  return data
}
