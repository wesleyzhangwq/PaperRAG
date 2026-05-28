<template>
  <div class="rounded-card border border-border bg-bg-secondary px-3 py-2">
    <button
      type="button"
      class="w-full flex items-center justify-between text-left"
      @click="open = !open"
    >
      <span class="text-xs text-text-tertiary flex items-center gap-2">
        <span v-if="running" class="inline-block w-1.5 h-1.5 rounded-full bg-accent animate-pulse"></span>
        <span v-else class="text-text-tertiary">💭</span>
        模型思考 · {{ reasoning.length }} 字
      </span>
      <span class="text-text-tertiary text-xs">{{ open ? '收起 ▴' : '展开 ▾' }}</span>
    </button>
    <div v-if="open" class="mt-2 max-h-64 overflow-y-auto reasoning-block">{{ reasoning }}</div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'

defineProps<{ reasoning: string; running: boolean }>()
const open = ref(false)  // collapsed by default — small, gray, foldable
</script>
