import { useChatStore } from '../stores/chat'
import { useConversationsStore } from '../stores/conversations'
import { useSSE } from './useSSE'
import type { Message, ThinkingStep, ToolCallEvent, ToolResultEvent } from '../types'

export function useChat() {
  const chat = useChatStore()
  const convs = useConversationsStore()
  const { streamChat, abort } = useSSE()

  async function sendMessage(query: string) {
    if (!query.trim() || chat.isLoading) return
    if (!convs.activeId) {
      await convs.createNew()
    }

    // 1) Append user message
    convs.appendMessage({
      id: crypto.randomUUID(),
      role: 'user',
      content: query,
      timestamp: Date.now(),
    })
    convs.bumpActive(query)

    // 2) Append empty assistant message that will be filled by streaming events
    const assistant: Message = {
      id: crypto.randomUUID(),
      role: 'assistant',
      content: '',
      reasoning: '',
      thinking: [],
      toolCalls: [],
      toolResults: [],
      sources: [],
      elapsedMs: 0,
      timestamp: Date.now(),
      pending: true,
    }
    convs.appendMessage(assistant)

    chat.isLoading = true
    chat.currentElapsedMs = 0
    let tokenBuffer = ''
    let tokenFlushTimer: ReturnType<typeof setInterval> | null = null

    function appendContent(text: string) {
      const cur = currentAssistant()
      if (!cur || !text) return
      convs.updateLastAssistant({ content: (cur.content || '') + text })
    }

    function flushTokenBuffer(force = false) {
      if (!tokenBuffer) return
      if (force) {
        const rest = tokenBuffer
        tokenBuffer = ''
        appendContent(rest)
        return
      }
      const sliceSize = tokenBuffer.length > 80 ? 8 : 4
      const next = tokenBuffer.slice(0, sliceSize)
      tokenBuffer = tokenBuffer.slice(sliceSize)
      appendContent(next)
    }

    function startTokenFlush() {
      if (tokenFlushTimer) return
      tokenFlushTimer = setInterval(() => {
        flushTokenBuffer(false)
        if (!tokenBuffer && tokenFlushTimer) {
          clearInterval(tokenFlushTimer)
          tokenFlushTimer = null
        }
      }, 24)
    }

    function stopTokenFlush() {
      if (!tokenFlushTimer) return
      clearInterval(tokenFlushTimer)
      tokenFlushTimer = null
    }

    try {
      for await (const event of streamChat(query, convs.activeId)) {
        switch (event.type) {
          case 'conversation':
            // backend echoes the id (could differ if server generated one)
            if (event.data.conversation_id && event.data.conversation_id !== convs.activeId) {
              await convs.selectConversation(event.data.conversation_id)
            }
            break

          case 'plan': {
            const steps: ThinkingStep[] = (event.data.steps || []).map((s, i) => ({
              index: i,
              action: s.action,
              reason: s.reason,
              status: 'pending',
            }))
            convs.updateLastAssistant({ thinking: steps })
            break
          }

          case 'step_start': {
            const cur = currentAssistant()
            if (!cur) break
            const steps = [...(cur.thinking || [])]
            const target = steps.find(s => s.index === event.data.index)
            if (target) target.status = 'running'
            convs.updateLastAssistant({ thinking: steps })
            break
          }

          case 'step_done': {
            const cur = currentAssistant()
            if (!cur) break
            const steps = [...(cur.thinking || [])]
            const running = steps.find(s => s.status === 'running')
              || steps.find(s => s.status === 'pending')
            if (running) {
              running.status = 'done'
              running.outputSummary = event.data.output_summary
              running.durationMs = event.data.duration_ms
            }
            convs.updateLastAssistant({ thinking: steps })
            break
          }

          case 'tool_call': {
            const cur = currentAssistant()
            if (!cur) break
            const toolCalls = [...(cur.toolCalls || []), event.data as ToolCallEvent]
            convs.updateLastAssistant({ toolCalls })
            break
          }

          case 'tool_result': {
            const cur = currentAssistant()
            if (!cur) break
            const toolResults = [...(cur.toolResults || []), event.data as ToolResultEvent]
            convs.updateLastAssistant({ toolResults })
            break
          }

          case 'reflection': {
            if (event.data.passed === false) {
              const cur = currentAssistant()
              if (!cur) break
              const steps = [...(cur.thinking || [])]
              const running = steps.find(s => s.status === 'running')
              if (running) running.status = 'failed'
              convs.updateLastAssistant({ thinking: steps })
            }
            break
          }

          case 're_plan': {
            const cur = currentAssistant()
            if (!cur) break
            const steps = [...(cur.thinking || [])]
            const extras = (event.data.new_steps as { action: string; reason: string }[] || []).map((s, i) => ({
              index: steps.length + i,
              action: s.action,
              reason: s.reason,
              status: 'pending' as const,
            }))
            convs.updateLastAssistant({ thinking: [...steps, ...extras] })
            break
          }

          case 'reasoning_token': {
            // Reasoning tokens can contain hidden prompt or chain-of-thought text.
            // Keep them out of the normal product UI.
            break
          }

          case 'token': {
            tokenBuffer += event.data.t
            startTokenFlush()
            break
          }

          case 'sources':
            flushTokenBuffer(true)
            convs.updateLastAssistant({ sources: event.data.sources })
            break

          case 'presentation':
            flushTokenBuffer(true)
            convs.updateLastAssistant({ presentation: event.data })
            break

          case 'elapsed':
            chat.currentElapsedMs = event.data.ms
            convs.updateLastAssistant({ elapsedMs: event.data.ms })
            break

          case 'done': {
            flushTokenBuffer(true)
            const cur = currentAssistant()
            if (cur) {
              const steps = (cur.thinking || []).map(s =>
                s.status === 'pending' ? { ...s, status: 'done' as const } : s
              )
              convs.updateLastAssistant({ thinking: steps, pending: false })
            }
            break
          }

          case 'error':
            flushTokenBuffer(true)
            convs.updateLastAssistant({
              content: `❌ 出错：${event.data.message}`,
              pending: false,
            })
            break
        }
      }
    } catch {
      tokenBuffer = ''
      stopTokenFlush()
      convs.updateLastAssistant({
        content: '❌ 连接中断，请重试。',
        pending: false,
      })
    } finally {
      flushTokenBuffer(true)
      stopTokenFlush()
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
