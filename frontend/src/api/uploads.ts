import api from './client'
import type { ArxivImportBatchResponse, UploadJob, UploadJobListResponse } from '../types'

export async function importArxivPapers(arxivIds: string[]): Promise<ArxivImportBatchResponse> {
  const { data } = await api.post<ArxivImportBatchResponse>('/upload/arxiv', {
    arxiv_ids: arxivIds,
  }, {
    timeout: 30000,
  })
  return data
}

export async function getUploadJob(jobId: string): Promise<UploadJob> {
  const { data } = await api.get<UploadJob>(`/upload/jobs/${jobId}`)
  return data
}

export async function listUploadJobs(params: { limit?: number; offset?: number } = {}): Promise<UploadJobListResponse> {
  const { data } = await api.get<UploadJobListResponse>('/upload/jobs', { params })
  return data
}
