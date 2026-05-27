<template>
  <div ref="listRef" class="flex-1 overflow-y-auto px-4 py-6">
    <div class="max-w-3xl mx-auto space-y-5">
      <div v-if="messages.length === 0" class="text-center py-20 text-text-tertiary text-sm">
        发送一条消息开始对话吧～
      </div>

      <template v-for="msg in messages" :key="msg.id">
        <UserBubble v-if="msg.role === 'user'" :content="msg.content" />
        <div v-else class="space-y-3">
          <!-- ① While streaming: show live progress card -->
          <ThinkingCard
            v-if="msg.pending"
            :steps="msg.thinking || []"
            :tool-calls="msg.toolCalls || []"
            :tool-results="msg.toolResults || []"
            :elapsed-ms="msg.elapsedMs || 0"
            :running="true"
          />
          <!-- Optional: reasoning trace (model's internal thinking, collapsed by default) -->
          <ReasoningBlock
            v-if="msg.reasoning && msg.reasoning.length > 0"
            :reasoning="msg.reasoning"
            :running="!!msg.pending"
          />
          <!-- ② Once content arrives (or presentation), show the structured AnswerCard -->
          <AnswerCard
            v-if="msg.content || msg.presentation || !msg.pending"
            :content="msg.content"
            :sources="msg.sources"
            :presentation="msg.presentation"
            :streaming="!!msg.pending"
          />
        </div>
      </template>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch, nextTick } from 'vue'
import type { Message } from '../../types'
import UserBubble from './UserBubble.vue'
import ThinkingCard from './ThinkingCard.vue'
import ReasoningBlock from './ReasoningBlock.vue'
import AnswerCard from '../answer/AnswerCard.vue'

const props = defineProps<{ messages: Message[] }>()

const listRef = ref<HTMLElement>()

watch(
  () => [props.messages.length, props.messages[props.messages.length - 1]?.content],
  async () => {
    await nextTick()
    if (listRef.value) listRef.value.scrollTop = listRef.value.scrollHeight
  },
  { immediate: true },
)
</script>
