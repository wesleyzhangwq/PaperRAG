<template>
  <div class="flex-1 flex flex-col overflow-hidden">
    <MessageList
      :messages="convs.activeMessages"
      :overview="overview"
      :overview-loading="overviewLoading"
      :overview-error="overviewError"
      @ask="useSuggestedQuestion"
    />
    <InputArea
      @send="chat.sendMessage"
      :disabled="chatState.isLoading"
      :draft="draftQuestion"
    />
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref, nextTick } from 'vue'
import { useChatStore } from '../stores/chat'
import { useConversationsStore } from '../stores/conversations'
import { useChat } from '../composables/useChat'
import { getCorpusOverview } from '../api/papers'
import MessageList from '../components/chat/MessageList.vue'
import InputArea from '../components/chat/InputArea.vue'
import type { CorpusOverviewResponse } from '../types'

const convs = useConversationsStore()
const chatState = useChatStore()
const chat = useChat()
const overview = ref<CorpusOverviewResponse | null>(null)
const overviewLoading = ref(false)
const overviewError = ref('')
const draftQuestion = ref('')

onMounted(loadOverview)

async function loadOverview() {
  overviewLoading.value = true
  overviewError.value = ''
  try {
    overview.value = await getCorpusOverview()
  } catch {
    overviewError.value = '语料概览暂时不可用'
  } finally {
    overviewLoading.value = false
  }
}

async function useSuggestedQuestion(question: string) {
  draftQuestion.value = ''
  await nextTick()
  draftQuestion.value = question
}
</script>
