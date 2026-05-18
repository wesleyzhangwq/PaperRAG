import { ref } from 'vue'
import type { SSEIntent, SSEPlan, SSEReflection, StepTrace, Source } from '../types'

export type SSEEvent =
  | { type: 'intent'; data: SSEIntent }
  | { type: 'plan'; data: SSEPlan }
  | { type: 'step_start'; data: { index: number; action: string; reason: string } }
  | { type: 'step_done'; data: StepTrace }
  | { type: 'reflection'; data: SSEReflection }
  | { type: 're_plan'; data: { new_steps: unknown[] } }
  | { type: 'token'; data: { t: string } }
  | { type: 'sources'; data: { sources: Source[] } }
  | { type: 'done'; data: { total_ms: number; steps_count: number; reflections: number } }
  | { type: 'error'; data: { message: string } }

export function useSSE() {
  const isConnected = ref(false)
  let abortController: AbortController | null = null

  async function* streamChat(query: string, sessionId: string): AsyncGenerator<SSEEvent> {
    abortController = new AbortController()
    isConnected.value = true

    const baseUrl = import.meta.env.VITE_API_BASE || 'http://localhost:8000'
    const response = await fetch(`${baseUrl}/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, session_id: sessionId }),
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
            eventType = line.slice(7)
          } else if (line.startsWith('data: ') && eventType) {
            try {
              const data = JSON.parse(line.slice(6))
              yield { type: eventType, data } as SSEEvent
            } catch { /* skip malformed */ }
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
