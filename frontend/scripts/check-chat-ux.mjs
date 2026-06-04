import { readFileSync } from 'node:fs'
import { join } from 'node:path'

const root = new URL('..', import.meta.url).pathname

function read(path) {
  return readFileSync(join(root, path), 'utf8')
}

function assert(condition, message) {
  if (!condition) {
    throw new Error(message)
  }
}

const markdown = read('src/utils/markdown.ts')
const messageList = read('src/components/chat/MessageList.vue')
const answerCard = read('src/components/answer/AnswerCard.vue')
const chatLayout = read('src/layouts/ChatLayout.vue')
const debugPanel = read('src/components/answer/DebugPanel.vue')
const thinkingCard = read('src/components/chat/ThinkingCard.vue')
const useChat = read('src/composables/useChat.ts')
const conversationsStore = read('src/stores/conversations.ts')
const duration = read('src/utils/duration.ts')
const synthesis = read('../backend/app/agent/nodes/synthesis.py')
const chatRouter = read('../backend/app/routers/chat.py')
const conversationsRouter = read('../backend/app/routers/conversations.py')

assert(
  markdown.includes('CITATION_TOKEN_PREFIX')
    && markdown.includes('tokens[token] = citationPill')
    && markdown.includes('html = md.render(processed)')
    && markdown.includes('html.split(token).join(pill)')
    && markdown.includes('stripHiddenReasoning')
    && !markdown.includes('citation-pill inline-flex'),
  'markdown citations must render safe HTML pills after markdown-it and hidden reasoning must not reach answer HTML',
)

assert(
  messageList.includes('isNearBottom') && messageList.includes('@scroll="handleScroll"'),
  'message list must only auto-stick when the user is already near the bottom',
)

assert(
  answerCard.includes('Worked for') && answerCard.includes('elapsedMs'),
  'completed answers must show a Worked for duration badge above the answer',
)

assert(
  chatLayout.includes('findReusableNewConversation') && chatLayout.includes('selectConversation(existing.id)'),
  'new chat button must reuse an existing Recents 新对话 instead of creating duplicates',
)

assert(
  debugPanel.includes('copyDebugDetails') && debugPanel.includes('navigator.clipboard.writeText'),
  'debug details must provide one-click copy',
)

assert(
  answerCard.includes('@click="onCitationClick"') && answerCard.includes('window.open'),
  'inline citation pills must react to click and open the cited source',
)

assert(
  conversationsStore.includes('deriveElapsedMs') && conversationsStore.includes('duration_ms'),
  'persisted answers must recover elapsedMs from saved step durations so Worked for survives reloads',
)

assert(
  duration.includes('formatStepDuration')
    && duration.includes('toFixed(2)')
    && !duration.includes('toFixed(1)'),
  'single execution step durations >= 1000ms must be displayed in seconds with two decimals',
)

assert(
  !answerCard.includes('ExecutionSteps') && answerCard.includes('DebugPanel'),
  'execution steps must not be duplicated after the answer; debug details should remain',
)

assert(
  thinkingCard.includes('watch(') && thinkingCard.includes('expanded.value = false'),
  'thinking card must automatically collapse after the run completes',
)

assert(
  useChat.includes('flushTokenBuffer')
    && useChat.includes("tokenBuffer = ''")
    && !useChat.includes("reasoning: (cur.reasoning || '') + event.data.t"),
  'chat streaming must buffer answer tokens smoothly and must not expose model reasoning text',
)

assert(
  !synthesis.includes('emit("reasoning_token"') && !synthesis.includes("emit('reasoning_token'"),
  'backend must not emit model reasoning tokens to the browser',
)

assert(
  chatRouter.includes('strip_hidden_reasoning') && conversationsRouter.includes('strip_hidden_reasoning'),
  'backend history APIs must filter persisted hidden reasoning before returning or reusing assistant messages',
)

console.log('chat UX checks passed')
