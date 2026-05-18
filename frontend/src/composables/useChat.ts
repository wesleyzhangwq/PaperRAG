import { useChatStore } from '../stores/chat'
import { useSSE } from './useSSE'
import { useThinking } from './useThinking'

export function useChat() {
  const store = useChatStore()
  const { streamChat, abort } = useSSE()
  const thinking = useThinking()

  async function sendMessage(query: string) {
    if (!query.trim() || store.isLoading) return

    store.addUserMessage(query)
    store.isLoading = true
    store.addAssistantMessage('')
    thinking.reset()

    try {
      for await (const event of streamChat(query, store.sessionId)) {
        switch (event.type) {
          case 'plan':
            thinking.startFromPlan(event.data)
            break
          case 'step_done':
            thinking.markStepDone(event.data)
            break
          case 'reflection':
            if (!event.data.passed) thinking.markFailed()
            break
          case 'token':
            store.updateLastAssistant(
              (store.messages[store.messages.length - 1]?.content || '') + event.data.t
            )
            break
          case 'sources':
            store.updateLastAssistant(
              store.messages[store.messages.length - 1]?.content || '',
              event.data.sources
            )
            break
          case 'done':
            thinking.isThinking.value = false
            break
          case 'error':
            store.updateLastAssistant(`Error: ${event.data.message}`)
            break
        }
      }
    } catch (e) {
      store.updateLastAssistant('连接中断，请重试。')
    } finally {
      store.isLoading = false
    }
  }

  return { sendMessage, abort, thinking }
}
