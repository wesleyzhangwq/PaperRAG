<template>
  <div class="border border-border rounded-card bg-bg-card p-3 hover:border-accent/40 transition">
    <div class="flex items-start gap-2">
      <span class="text-xs text-text-tertiary font-mono mt-0.5 flex-shrink-0">{{ index }}.</span>
      <div class="flex-1 min-w-0">
        <div class="flex items-start justify-between gap-2">
          <a
            v-if="source.arxiv_url"
            :href="source.arxiv_url || undefined"
            target="_blank"
            rel="noopener"
            class="text-sm font-medium text-text-primary hover:text-accent leading-snug line-clamp-2"
          >{{ source.title }}</a>
          <span
            v-else
            class="text-sm font-medium text-text-primary leading-snug line-clamp-2"
          >{{ source.title }}</span>
          <RelevancePill :level="source.relevance" />
        </div>
        <div class="mt-1 flex items-center gap-2 text-xs text-text-tertiary">
          <code class="font-mono">{{ source.paper_id }}</code>
          <span v-if="source.year">· {{ source.year }}</span>
          <span v-if="source.primary_category">· {{ source.primary_category }}</span>
          <span v-if="source.hit_count">· 命中 {{ source.hit_count }} 段</span>
        </div>
        <p v-if="source.summary" class="mt-1.5 text-xs text-text-secondary line-clamp-2">
          {{ source.summary }}
        </p>
        <button
          v-if="source.snippets.length > 0"
          @click="open = !open"
          class="mt-1.5 text-xs text-text-tertiary hover:text-accent"
        >
          {{ open ? '收起命中片段 ▴' : `展开命中片段 (${source.snippets.length}) ▾` }}
        </button>
        <div v-if="open" class="mt-2 space-y-1.5">
          <blockquote
            v-for="(sn, i) in source.snippets"
            :key="i"
            class="text-xs text-text-secondary border-l-2 border-accent/40 pl-2 py-0.5 italic"
          >"{{ sn }}"</blockquote>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import type { SourceCard } from '../../types'
import RelevancePill from './RelevancePill.vue'

defineProps<{ source: SourceCard; index: number }>()
const open = ref(false)
</script>
