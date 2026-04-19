import { defineStore } from 'pinia'
import { ref } from 'vue'
import { chat, type ChatFilter, type Source } from '../api/client'

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
  used_chunks?: number
  created_at: number
}

export const useChatStore = defineStore('chat', () => {
  const messages = ref<ChatMessage[]>([])
  const loading = ref(false)
  const error = ref<string | null>(null)
  const currentSources = ref<Source[]>([])

  async function ask(query: string, filter?: ChatFilter) {
    if (!query.trim()) return
    error.value = null
    const userId = crypto.randomUUID()
    messages.value.push({
      id: userId,
      role: 'user',
      content: query,
      created_at: Date.now(),
    })
    loading.value = true
    try {
      const resp = await chat(query, filter)
      messages.value.push({
        id: crypto.randomUUID(),
        role: 'assistant',
        content: resp.answer,
        sources: resp.sources,
        used_chunks: resp.used_chunks,
        created_at: Date.now(),
      })
      currentSources.value = resp.sources
    } catch (e: any) {
      error.value = e?.response?.data?.detail ?? e?.message ?? 'unknown error'
      messages.value.push({
        id: crypto.randomUUID(),
        role: 'assistant',
        content: `请求失败：${error.value}`,
        created_at: Date.now(),
      })
    } finally {
      loading.value = false
    }
  }

  function clear() {
    messages.value = []
    currentSources.value = []
    error.value = null
  }

  return { messages, loading, error, currentSources, ask, clear }
})
