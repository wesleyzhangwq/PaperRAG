<template>
  <div class="h-full flex bg-bg-primary">
    <aside v-if="sidebarOpen" class="w-64 border-r border-border bg-bg-secondary flex flex-col">
      <div class="p-4 border-b border-border">
        <h1 class="text-lg font-semibold text-text-primary">PaperRAG</h1>
      </div>
      <div class="p-3">
        <button @click="newChat" class="w-full px-3 py-2 rounded-sm text-sm text-text-secondary hover:bg-bg-primary transition">
          + New Conversation
        </button>
      </div>
    </aside>
    <main class="flex-1 flex flex-col min-w-0">
      <header class="h-12 border-b border-border flex items-center px-4 bg-bg-card">
        <button @click="sidebarOpen = !sidebarOpen" class="mr-3 text-text-tertiary hover:text-text-primary">
          ☰
        </button>
        <span class="text-sm text-text-secondary">Agentic RAG Paper Assistant</span>
      </header>
      <ChatView />
    </main>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import ChatView from '../views/ChatView.vue'
import { useChatStore } from '../stores/chat'

const sidebarOpen = ref(true)
const store = useChatStore()

function newChat() {
  store.newConversation()
}
</script>
