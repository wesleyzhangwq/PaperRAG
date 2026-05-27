<template>
  <section class="space-y-1">
    <header class="px-3 flex items-center justify-between">
      <h3 class="text-[13px] text-text-tertiary font-normal">{{ title }}</h3>
      <slot name="header-right" />
    </header>
    <div v-if="items.length === 0 && emptyText" class="px-3 py-1 text-xs text-text-tertiary">
      {{ emptyText }}
    </div>
    <ul class="space-y-0.5">
      <li v-for="conv in items" :key="conv.id">
        <div
          :class="[
            'group flex items-center gap-2.5 px-3 py-1.5 rounded-md cursor-pointer transition-colors',
            activeId === conv.id ? 'bg-bg-hover' : 'hover:bg-bg-hover/60',
          ]"
          @click="$emit('select', conv.id)"
        >
          <!-- Status indicator dot on the left (matches screenshot) -->
          <span
            class="flex-shrink-0 w-2 h-2 rounded-full border"
            :class="dotClass(conv)"
          ></span>
          <!-- Title -->
          <span
            class="flex-1 text-[14px] truncate"
            :class="activeId === conv.id ? 'text-text-primary font-medium' : 'text-text-secondary'"
            :title="conv.title"
          >{{ conv.title || '新对话' }}</span>
          <!-- Hover action menu -->
          <div
            class="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition"
            @click.stop
          >
            <button
              @click="$emit('toggle-pin', conv.id)"
              :title="conv.pinned ? '取消置顶' : '置顶'"
              class="p-1 rounded hover:bg-bg-card text-text-tertiary hover:text-accent"
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 17v5" />
                <path d="M9 10.76V6h6v4.76l3.39 3.39c.39.39.39 1.02 0 1.41-.18.18-.43.29-.7.29H6.31c-.55 0-1-.45-1-1 0-.27.11-.52.29-.7L9 10.76z" />
              </svg>
            </button>
            <button
              @click="onRename(conv)"
              title="重命名"
              class="p-1 rounded hover:bg-bg-card text-text-tertiary hover:text-accent"
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 20h9" />
                <path d="M16.5 3.5a2.121 2.121 0 113 3L7 19l-4 1 1-4 12.5-12.5z" />
              </svg>
            </button>
            <button
              @click="$emit('remove', conv.id)"
              title="删除"
              class="p-1 rounded hover:bg-bg-card text-text-tertiary hover:text-bad"
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="3 6 5 6 21 6" />
                <path d="M19 6l-2 14a2 2 0 01-2 2H9a2 2 0 01-2-2L5 6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2" />
              </svg>
            </button>
          </div>
        </div>
      </li>
    </ul>
  </section>
</template>

<script setup lang="ts">
import type { Conversation } from '../../types'

const props = defineProps<{
  title: string
  items: Conversation[]
  activeId: string
  emptyText?: string
}>()

const emit = defineEmits<{
  select: [id: string]
  'toggle-pin': [id: string]
  remove: [id: string]
  rename: [id: string, title: string]
}>()

function dotClass(conv: Conversation): string {
  if (props.activeId === conv.id) {
    // Filled accent dot for the active conversation
    return 'bg-accent border-accent'
  }
  if (conv.pinned) {
    return 'bg-text-tertiary/30 border-text-tertiary/50'
  }
  return 'bg-transparent border-text-tertiary/40'
}

function onRename(conv: Conversation) {
  const title = window.prompt('重命名对话', conv.title)
  if (title && title.trim()) emit('rename', conv.id, title.trim())
}
</script>
