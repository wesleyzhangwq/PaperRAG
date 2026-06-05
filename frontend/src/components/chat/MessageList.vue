<template>
  <div ref="listRef" class="flex-1 overflow-y-auto px-4 py-6" @scroll="handleScroll">
    <div class="max-w-3xl mx-auto space-y-5">
      <CorpusOverviewCard
        v-if="messages.length === 0"
        :overview="overview"
        :loading="overviewLoading"
        :error="overviewError"
        @ask="$emit('ask', $event)"
      />

      <template v-for="msg in messages" :key="msg.id">
        <UserBubble v-if="msg.role === 'user'" :content="msg.content" />
        <div v-else class="space-y-3">
          <!-- ① While streaming: show live progress card -->
          <ThinkingCard
            v-if="hasThinking(msg)"
            :steps="msg.thinking || []"
            :tool-calls="msg.toolCalls || []"
            :tool-results="msg.toolResults || []"
            :elapsed-ms="msg.elapsedMs || 0"
            :running="!!msg.pending"
          />
          <!-- ② Once content arrives (or presentation), show the structured AnswerCard -->
          <AnswerCard
            v-if="msg.content || msg.presentation || !msg.pending"
            :content="msg.content"
            :sources="msg.sources"
            :presentation="msg.presentation"
            :streaming="!!msg.pending"
            :elapsed-ms="msg.elapsedMs || 0"
            :conversation-id="convs.activeId"
            :message-id="serverMessageId(msg.id)"
          />
        </div>
      </template>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch, nextTick } from 'vue'
import type { CorpusOverviewResponse, Message } from '../../types'
import { useConversationsStore } from '../../stores/conversations'
import UserBubble from './UserBubble.vue'
import ThinkingCard from './ThinkingCard.vue'
import AnswerCard from '../answer/AnswerCard.vue'
import CorpusOverviewCard from './CorpusOverviewCard.vue'

const props = defineProps<{
  messages: Message[]
  overview: CorpusOverviewResponse | null
  overviewLoading: boolean
  overviewError: string
}>()
defineEmits<{ ask: [question: string] }>()
const convs = useConversationsStore()

const listRef = ref<HTMLElement>()
const stickToBottom = ref(true)
const BOTTOM_THRESHOLD_PX = 96

watch(
  () => {
    const last = props.messages[props.messages.length - 1]
    return [
      props.messages.length,
      last?.content,
      last?.pending,
      last?.elapsedMs,
      last?.thinking?.length,
      last?.presentation,
    ]
  },
  async () => {
    await nextTick()
    if (stickToBottom.value) scrollToBottom()
  },
  { immediate: true },
)

function handleScroll() {
  stickToBottom.value = isNearBottom()
}

function isNearBottom(): boolean {
  const el = listRef.value
  if (!el) return true
  return el.scrollHeight - el.scrollTop - el.clientHeight <= BOTTOM_THRESHOLD_PX
}

function scrollToBottom() {
  const el = listRef.value
  if (!el) return
  el.scrollTop = el.scrollHeight
}

function hasThinking(msg: Message): boolean {
  return msg.role === 'assistant' && !!(msg.pending || msg.thinking?.length)
}

function serverMessageId(id: string): number | undefined {
  if (!id.startsWith('srv-')) return undefined
  const parsed = Number(id.slice(4))
  return Number.isFinite(parsed) ? parsed : undefined
}
</script>
