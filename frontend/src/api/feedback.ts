import api from './client'

export async function submitAnswerFeedback(payload: {
  conversation_id: string
  message_id?: number
  vote: 'up' | 'down'
  reason?: string
  comment?: string
}): Promise<void> {
  await api.post('/feedback', payload)
}
