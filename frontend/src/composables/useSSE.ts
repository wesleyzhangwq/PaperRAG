import { ref } from 'vue'
import type { SSEPlan, SSEStage, Source, Presentation } from '../types'

export type SSEEvent =
  | { type: 'conversation'; data: { conversation_id: string } }
  | { type: 'stage'; data: SSEStage }
  | { type: 'plan'; data: SSEPlan }
  | { type: 'answer_start'; data: { attempt: number; reset?: boolean } }
  | { type: 'token'; data: { t: string } }
  | { type: 'sources'; data: { sources: Source[] } }
  | { type: 'presentation'; data: Presentation }
  | { type: 'elapsed'; data: { ms: number; final?: boolean } }
  | { type: 'done'; data: { steps_count: number; reflections: number } }
  | { type: 'error'; data: { message: string } }

/**
 * Spec-compliant SSE client over fetch streaming.
 *
 * Parser rules (https://html.spec.whatwg.org/multipage/server-sent-events.html):
 * - events are separated by blank lines; fields accumulate until then
 * - multiple `data:` lines join with "\n"
 * - tolerate CRLF line endings and `: comment` keep-alives
 */
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
    let response: Response
    try {
      response = await fetch(`${baseUrl}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          conversation_id: conversationId,
          session_id: conversationId,
        }),
        signal: abortController.signal,
      })
    } catch (e) {
      isConnected.value = false
      yield { type: 'error', data: { message: e instanceof Error ? e.message : '网络连接失败' } }
      return
    }

    if (!response.ok || !response.body) {
      isConnected.value = false
      yield { type: 'error', data: { message: `HTTP ${response.status}` } }
      return
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    let eventType = ''
    let dataLines: string[] = []

    function* dispatch(): Generator<SSEEvent> {
      if (!eventType && dataLines.length === 0) return
      const raw = dataLines.join('\n')
      const type = eventType || 'message'
      eventType = ''
      dataLines = []
      if (!raw) return
      try {
        yield { type, data: JSON.parse(raw) } as SSEEvent
      } catch {
        /* malformed frame — skip */
      }
    }

    try {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        // Tolerate \n and \r\n line endings.
        const lines = buffer.split(/\r?\n/)
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line === '') {
            yield* dispatch()
          } else if (line.startsWith(':')) {
            continue // comment / keep-alive
          } else if (line.startsWith('event:')) {
            eventType = line.slice(6).trim()
          } else if (line.startsWith('data:')) {
            dataLines.push(line.slice(5).trimStart())
          }
        }
      }
      yield* dispatch() // flush trailing frame without final blank line
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
