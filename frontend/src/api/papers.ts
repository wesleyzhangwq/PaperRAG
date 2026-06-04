import api from './client'
import type { PaperListResponse } from '../types'

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
