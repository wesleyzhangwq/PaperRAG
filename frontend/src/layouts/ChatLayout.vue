<template>
  <div class="h-full flex bg-bg-primary text-text-primary">
    <!-- ============== Sidebar ============== -->
    <aside
      v-if="sidebarOpen"
      class="w-72 border-r border-border bg-bg-secondary flex flex-col flex-shrink-0"
    >
      <!-- Brand + new chat -->
      <div class="px-4 py-3 flex items-center justify-between">
        <h1 class="text-[15px] font-semibold text-text-primary tracking-tight">PaperRAG</h1>
        <button
          @click="newChat"
          title="新建对话"
          class="p-1.5 rounded-md text-text-secondary hover:text-accent hover:bg-bg-hover transition"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 20h9" />
            <path d="M16.5 3.5a2.121 2.121 0 013 3L7 19l-4 1 1-4 12.5-12.5z" />
          </svg>
        </button>
      </div>

      <!-- Conversations list -->
      <div class="flex-1 overflow-y-auto px-2 pb-3 space-y-5">
        <ConversationSection
          v-if="convs.pinned.length > 0"
          title="Pinned"
          :items="convs.pinned"
          :active-id="convs.activeId"
          @select="onSelect"
          @toggle-pin="onTogglePin"
          @remove="onRemove"
          @rename="onRename"
        />
        <ConversationSection
          title="Recents"
          :items="convs.recent"
          :active-id="convs.activeId"
          empty-text="还没有对话"
          @select="onSelect"
          @toggle-pin="onTogglePin"
          @remove="onRemove"
          @rename="onRename"
        >
          <template #header-right>
            <button
              title="排序与筛选（占位）"
              class="p-1 rounded text-text-tertiary hover:text-text-secondary hover:bg-bg-hover transition"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <line x1="4" y1="6" x2="14" y2="6" />
                <line x1="10" y1="12" x2="20" y2="12" />
                <line x1="4" y1="18" x2="14" y2="18" />
                <circle cx="17" cy="6" r="1.5" fill="currentColor" />
                <circle cx="7" cy="12" r="1.5" fill="currentColor" />
                <circle cx="17" cy="18" r="1.5" fill="currentColor" />
              </svg>
            </button>
          </template>
        </ConversationSection>
      </div>

      <!-- Footer: theme toggle -->
      <div class="px-4 py-3 border-t border-border flex items-center justify-between">
        <span class="text-xs text-text-tertiary">{{ themeLabel }}</span>
        <button
          @click="theme.toggle"
          class="px-2 py-1 rounded-md text-sm hover:bg-bg-hover transition"
          :title="theme.theme === 'light' ? '切换到夜间' : '切换到日间'"
        >
          <span v-if="theme.theme === 'light'">🌙</span>
          <span v-else>☀️</span>
        </button>
      </div>
    </aside>

    <!-- ============== Main ============== -->
    <main class="flex-1 flex flex-col min-w-0">
      <header class="h-12 border-b border-border flex items-center px-4 bg-bg-card">
        <button
          @click="sidebarOpen = !sidebarOpen"
          class="mr-3 text-text-tertiary hover:text-text-primary"
          :title="sidebarOpen ? '收起侧栏' : '展开侧栏'"
        >☰</button>
        <span class="text-sm text-text-secondary truncate">
          {{ convs.active?.title || 'Agentic RAG Paper Assistant' }}
        </span>
      </header>
      <ChatView />
    </main>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import ChatView from '../views/ChatView.vue'
import ConversationSection from '../components/sidebar/ConversationSection.vue'
import { useConversationsStore } from '../stores/conversations'
import { useThemeStore } from '../stores/theme'

const convs = useConversationsStore()
const theme = useThemeStore()
const sidebarOpen = ref(true)

const themeLabel = computed(() =>
  theme.theme === 'light' ? '日间模式' : '夜间模式'
)

onMounted(async () => {
  await convs.loadAll()
})

async function newChat() {
  await convs.createNew()
}

async function onSelect(id: string) {
  await convs.selectConversation(id)
}
async function onTogglePin(id: string) {
  await convs.togglePin(id)
}
async function onRemove(id: string) {
  if (confirm('删除这个对话？此操作不可撤销。')) {
    await convs.remove(id)
  }
}
async function onRename(id: string, title: string) {
  await convs.rename(id, title)
}
</script>
