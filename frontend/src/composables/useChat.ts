import { useChatStore } from '../stores/chat'
import { useConversationsStore } from '../stores/conversations'
import { useSSE } from './useSSE'
import { applyPlanEvent, applyStageEvent } from '../utils/timeline'
import type { Message } from '../types'

/**
 * Streaming chat orchestration.
 *
 * Design notes:
 * - Stage/plan events carry STABLE ids; the timeline is a pure upsert — no
 *   index guessing, no plan/trace reconciliation heuristics.
 * - Tokens render at network speed: appended into a pending buffer and
 *   flushed once per animation frame (zero artificial latency, max one
 *   DOM update per frame).
 */
export function useChat() {
  const chat = useChatStore()
  const convs = useConversationsStore()
  const { streamChat, abort } = useSSE()

  async function sendMessage(query: string) {
    if (!query.trim() || chat.isLoading) return
    if (!convs.activeId) {
      await convs.createNew()
    }

    convs.appendMessage({
      id: crypto.randomUUID(),
      role: 'user',
      content: query,
      timestamp: Date.now(),
    })
    convs.bumpActive(query)

    const assistant: Message = {
      id: crypto.randomUUID(),
      role: 'assistant',
      content: '',
      timeline: [],
      sources: [],
      elapsedMs: 0,
      timestamp: Date.now(),
      pending: true,
      answering: false,
    }
    convs.appendMessage(assistant)

    chat.isLoading = true
    chat.currentElapsedMs = 0

    // --- rAF-batched token rendering -------------------------------------
    let tokenPending = ''
    let rafId: number | null = null

    function flushTokens() {
      rafId = null
      if (!tokenPending) return
      const text = tokenPending
      tokenPending = ''
      const cur = currentAssistant()
      if (cur) convs.updateLastAssistant({ content: (cur.content || '') + text })
    }

    function pushToken(t: string) {
      tokenPending += t
      if (rafId === null) rafId = requestAnimationFrame(flushTokens)
    }

    function flushTokensNow() {
      if (rafId !== null) {
        cancelAnimationFrame(rafId)
        rafId = null
      }
      if (tokenPending) {
        const text = tokenPending
        tokenPending = ''
        const cur = currentAssistant()
        if (cur) convs.updateLastAssistant({ content: (cur.content || '') + text })
      }
    }
    // ----------------------------------------------------------------------

    try {
      for await (const event of streamChat(query, convs.activeId)) {
        switch (event.type) {
          case 'conversation':
            if (event.data.conversation_id && event.data.conversation_id !== convs.activeId) {
              await convs.selectConversation(event.data.conversation_id)
            }
            break

          case 'stage': {
            const cur = currentAssistant()
            if (!cur) break
            convs.updateLastAssistant({
              timeline: applyStageEvent(cur.timeline || [], event.data),
            })
            break
          }

          case 'plan': {
            const cur = currentAssistant()
            if (!cur) break
            convs.updateLastAssistant({
              timeline: applyPlanEvent(cur.timeline || [], event.data),
            })
            break
          }

          case 'answer_start':
            flushTokensNow()
            // Re-generation replaces the previous attempt's text.
            convs.updateLastAssistant({ content: '', answering: true })
            break

          case 'token':
            pushToken(event.data.t)
            break

          case 'sources':
            flushTokensNow()
            convs.updateLastAssistant({ sources: event.data.sources })
            break

          case 'presentation':
            flushTokensNow()
            convs.updateLastAssistant({
              presentation: event.data,
              content: event.data.answer || currentAssistant()?.content || '',
            })
            break

          case 'elapsed':
            chat.currentElapsedMs = event.data.ms
            convs.updateLastAssistant({ elapsedMs: event.data.ms })
            break

          case 'done': {
            flushTokensNow()
            const cur = currentAssistant()
            if (cur) {
              // Anything still pending/running at done is finished.
              const timeline = (cur.timeline || []).map(item =>
                item.status === 'pending' || item.status === 'running'
                  ? { ...item, status: 'done' as const }
                  : item
              )
              convs.updateLastAssistant({ timeline, pending: false })
            }
            break
          }

          case 'error': {
            flushTokensNow()
            const cur = currentAssistant()
            const existing = cur?.content || ''
            convs.updateLastAssistant({
              content: existing || `❌ 出错：${event.data.message}`,
              pending: false,
            })
            break
          }
        }
      }
    } catch {
      flushTokensNow()
      convs.updateLastAssistant({
        content: currentAssistant()?.content || '❌ 连接中断，请重试。',
        pending: false,
      })
    } finally {
      flushTokensNow()
      const cur = currentAssistant()
      if (cur?.pending) convs.updateLastAssistant({ pending: false })
      chat.isLoading = false
    }
  }

  function currentAssistant(): Message | null {
    const arr = convs.activeMessages
    if (!arr.length) return null
    const last = arr[arr.length - 1]
    return last.role === 'assistant' ? last : null
  }

  return { sendMessage, abort }
}
