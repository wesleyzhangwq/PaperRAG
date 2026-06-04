import api from './client'
import type { UploadJob, UploadJobListResponse, UploadResponse } from '../types'

export async function uploadPdf(file: File, title?: string): Promise<UploadResponse> {
  const form = new FormData()
  form.append('file', file)
  if (title?.trim()) form.append('title', title.trim())
  const { data } = await api.post<UploadResponse>('/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 120000,
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
