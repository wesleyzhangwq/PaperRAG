import api from './client'
import type { Conversation, ServerMessage } from '../types'

export async function listConversations(): Promise<Conversation[]> {
  const { data } = await api.get<Conversation[]>('/conversations')
  return data
}

export async function createConversation(id?: string, title?: string): Promise<Conversation> {
  const { data } = await api.post<Conversation>('/conversations', { id, title })
  return data
}

export async function updateConversation(
  id: string,
  patch: { title?: string; pinned?: boolean }
): Promise<Conversation> {
  const { data } = await api.patch<Conversation>(`/conversations/${id}`, patch)
  return data
}

export async function deleteConversation(id: string): Promise<void> {
  await api.delete(`/conversations/${id}`)
}

export async function getMessages(id: string): Promise<ServerMessage[]> {
  const { data } = await api.get<ServerMessage[]>(`/conversations/${id}/messages`)
  return data
}
