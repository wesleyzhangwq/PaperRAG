import { ref } from 'vue'
import type {
  SSEIntent, SSEPlan, SSEReflection, StepTrace, Source,
  ToolCallEvent, ToolResultEvent, Presentation,
} from '../types'

export type SSEEvent =
  | { type: 'conversation'; data: { conversation_id: string } }
  | { type: 'intent'; data: SSEIntent }
  | { type: 'plan'; data: SSEPlan }
  | { type: 'step_start'; data: { index: number; action: string; reason: string } }
  | { type: 'step_done'; data: StepTrace }
  | { type: 'tool_call'; data: ToolCallEvent }
  | { type: 'tool_result'; data: ToolResultEvent }
  | { type: 'reflection'; data: SSEReflection }
  | { type: 're_plan'; data: { new_steps: unknown[] } }
  | { type: 'reasoning_token'; data: { t: string } }
  | { type: 'token'; data: { t: string } }
  | { type: 'sources'; data: { sources: Source[] } }
  | { type: 'presentation'; data: Presentation }
  | { type: 'elapsed'; data: { ms: number; final?: boolean } }
  | { type: 'done'; data: { steps_count: number; reflections: number } }
  | { type: 'error'; data: { message: string } }

export function useSSE() {
  const isConnected = ref(false)
  let abortController: AbortController | null = null

  async function* streamChat(
    query: string,
    conversationId: string,
  ): AsyncGenerator<SSEEvent> {
    abortController = new AbortController()
    isConnected.value = true

    const baseUrl = import.meta.env.VITE_API_BASE || 'http://localhost:8000'
    const response = await fetch(`${baseUrl}/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        conversation_id: conversationId,
        session_id: conversationId,
      }),
      signal: abortController.signal,
    })

    if (!response.ok || !response.body) {
      isConnected.value = false
      yield { type: 'error', data: { message: `HTTP ${response.status}` } }
      return
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    try {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        let eventType = ''
        for (const line of lines) {
          if (line.startsWith('event: ')) {
            eventType = line.slice(7).trim()
          } else if (line.startsWith('data: ') && eventType) {
            try {
              const data = JSON.parse(line.slice(6))
              yield { type: eventType, data } as SSEEvent
            } catch { /* skip malformed */ }
            eventType = ''
          } else if (line === '') {
            // event boundary
            eventType = ''
          }
        }
      }
    } finally {
      isConnected.value = false
    }
  }

  function abort() {
    abortController?.abort()
    isConnected.value = false
  }

  return { streamChat, abort, isConnected }
}
