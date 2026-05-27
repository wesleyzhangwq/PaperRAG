<template>
  <div class="flex-1 flex flex-col overflow-hidden">
    <MessageList :messages="convs.activeMessages" />
    <InputArea @send="chat.sendMessage" :disabled="chatState.isLoading" />
  </div>
</template>

<script setup lang="ts">
import { useChatStore } from '../stores/chat'
import { useConversationsStore } from '../stores/conversations'
import { useChat } from '../composables/useChat'
import MessageList from '../components/chat/MessageList.vue'
import InputArea from '../components/chat/InputArea.vue'

const convs = useConversationsStore()
const chatState = useChatStore()
const chat = useChat()
</script>
