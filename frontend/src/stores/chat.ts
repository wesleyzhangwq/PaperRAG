import { defineStore } from 'pinia'
import { ref } from 'vue'
import type { Message, Source } from '../types'

export const useChatStore = defineStore('chat', () => {
  const messages = ref<Message[]>([])
  const sessionId = ref(crypto.randomUUID())
  const isLoading = ref(false)

  function addUserMessage(content: string) {
    messages.value.push({
      id: crypto.randomUUID(),
      role: 'user',
      content,
      timestamp: Date.now(),
    })
  }

  function addAssistantMessage(content: string, sources?: Source[]) {
    messages.value.push({
      id: crypto.randomUUID(),
      role: 'assistant',
      content,
      sources,
      timestamp: Date.now(),
    })
  }

  function updateLastAssistant(content: string, sources?: Source[]) {
    const last = messages.value[messages.value.length - 1]
    if (last && last.role === 'assistant') {
      last.content = content
      if (sources) last.sources = sources
    }
  }

  function newConversation() {
    messages.value = []
    sessionId.value = crypto.randomUUID()
  }

  return { messages, sessionId, isLoading, addUserMessage, addAssistantMessage, updateLastAssistant, newConversation }
})
