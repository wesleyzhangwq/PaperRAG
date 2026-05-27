<template>
  <div class="rounded-card border border-border bg-bg-secondary px-3 py-2.5">
    <div class="flex items-center gap-2 text-xs font-medium text-text-secondary mb-1">
      <span>🔍 检索概况</span>
      <span class="text-text-tertiary font-normal">
        · {{ summary.total_chunks }} 个片段 · {{ summary.total_papers }} 篇论文
      </span>
      <span
        v-if="summary.is_fallback"
        class="ml-auto text-warn text-[10px] uppercase tracking-wide"
      >扩展检索</span>
    </div>
    <p class="text-sm text-text-secondary leading-relaxed">{{ summary.narrative }}</p>
    <div v-if="summary.main_topics.length > 0" class="mt-1.5 flex flex-wrap gap-1">
      <span
        v-for="t in summary.main_topics"
        :key="t"
        class="text-[10px] px-1.5 py-0.5 rounded bg-bg-card text-text-tertiary font-mono"
      >{{ t }}</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { RetrievalSummary } from '../../types'
defineProps<{ summary: RetrievalSummary }>()
</script>
