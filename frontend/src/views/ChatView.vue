<template>
  <div class="flex-1 flex flex-col overflow-hidden">
    <MessageList :messages="store.messages" :thinking="chat.thinking" />
    <InputArea @send="chat.sendMessage" :disabled="store.isLoading" />
  </div>
</template>

<script setup lang="ts">
import { useChatStore } from '../stores/chat'
import { useChat } from '../composables/useChat'
import MessageList from '../components/chat/MessageList.vue'
import InputArea from '../components/chat/InputArea.vue'

const store = useChatStore()
const chat = useChat()
</script>
