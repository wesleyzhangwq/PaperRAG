<template>
  <div ref="listRef" class="flex-1 overflow-y-auto px-4 py-6 space-y-4">
    <template v-for="(msg, idx) in messages" :key="msg.id">
      <UserBubble v-if="msg.role === 'user'" :content="msg.content" />
      <template v-else>
        <ThinkingCard v-if="idx === messages.length - 1 && (thinking.isThinking.value || thinking.steps.value.length > 0)" :steps="thinking.steps.value" />
        <AssistantBubble :content="msg.content" :sources="msg.sources" />
      </template>
    </template>
  </div>
</template>

<script setup lang="ts">
import { ref, watchEffect, nextTick } from 'vue'
import type { Message, ThinkingStep } from '../../types'
import UserBubble from './UserBubble.vue'
import AssistantBubble from './AssistantBubble.vue'
import ThinkingCard from './ThinkingCard.vue'

const props = defineProps<{
  messages: Message[]
  thinking: { steps: { value: ThinkingStep[] }; isThinking: { value: boolean } }
}>()

const listRef = ref<HTMLElement>()

watchEffect(async () => {
  props.messages.length
  await nextTick()
  if (listRef.value) {
    listRef.value.scrollTop = listRef.value.scrollHeight
  }
})
</script>
