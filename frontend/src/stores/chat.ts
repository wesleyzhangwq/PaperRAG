import { defineStore } from 'pinia'
import { ref } from 'vue'
import { chatStream, type ChatFilter, type Source } from '../api/client'

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
  const streaming = ref(false)
  const error = ref<string | null>(null)
  const currentSources = ref<Source[]>([])
  const sessionId = ref(crypto.randomUUID())

  async function ask(query: string, filter?: ChatFilter) {
    if (!query.trim()) return
    error.value = null
    messages.value.push({
      id: crypto.randomUUID(),
      role: 'user',
      content: query,
      created_at: Date.now(),
    })
    loading.value = true

    const assistantId = crypto.randomUUID()
    messages.value.push({
      id: assistantId,
      role: 'assistant',
      content: '',
      created_at: Date.now(),
    })
    const assistantMsg = messages.value[messages.value.length - 1]

    try {
      await chatStream(
        query,
        filter,
        sessionId.value,
        (token) => {
          if (!streaming.value) {
            streaming.value = true
            loading.value = false
          }
          assistantMsg.content += token
        },
        (sources) => {
          assistantMsg.sources = sources
          currentSources.value = sources
        },
        () => {
          streaming.value = false
        },
        (err) => {
          error.value = err.message
          assistantMsg.content = `请求失败：${err.message}`
          streaming.value = false
          loading.value = false
        },
      )
    } catch (e: any) {
      error.value = e?.message ?? 'unknown error'
      assistantMsg.content = `请求失败：${error.value}`
    } finally {
      loading.value = false
      streaming.value = false
    }
  }

  function newConversation() {
    messages.value = []
    currentSources.value = []
    error.value = null
    sessionId.value = crypto.randomUUID()
  }

  return { messages, loading, streaming, error, currentSources, sessionId, ask, newConversation }
})
