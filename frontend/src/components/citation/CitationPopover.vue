<template>
  <div class="relative inline-block" @mouseenter="show = true" @mouseleave="show = false">
    <span
      class="citation-pill inline-flex items-center px-1.5 py-0.5 rounded text-xs font-medium bg-amber-100 text-amber-700 cursor-pointer hover:bg-amber-200 transition"
      >[{{ index }}]</span
    >
    <Transition
      enter-active-class="transition duration-150 ease-out"
      enter-from-class="opacity-0 translate-y-1"
      enter-to-class="opacity-100 translate-y-0"
      leave-active-class="transition duration-100 ease-in"
      leave-from-class="opacity-100 translate-y-0"
      leave-to-class="opacity-0 translate-y-1"
    >
      <div
        v-if="show && source"
        class="absolute z-50 bottom-full left-1/2 -translate-x-1/2 mb-2 w-72 bg-bg-card border border-border rounded-lg shadow-lg p-3"
      >
        <p class="text-sm font-medium text-text-primary leading-tight mb-1">{{ source.title }}</p>
        <p class="text-xs text-text-secondary mb-1">
          {{ source.authors?.join(', ') || 'Unknown' }}
          <span v-if="source.year"> · {{ source.year }}</span>
          <span v-if="source.primary_category"> · {{ source.primary_category }}</span>
        </p>
        <p v-if="source.score" class="text-xs text-text-tertiary mb-1">
          Score: {{ source.score.toFixed(2) }}
          <span v-if="source.page_num"> · Page {{ source.page_num }}</span>
        </p>
        <p v-if="source.snippet" class="text-xs text-text-secondary italic border-t border-border pt-1.5 mt-1.5 line-clamp-3">
          "{{ source.snippet }}"
        </p>
        <a
          v-if="source.arxiv_url"
          :href="source.arxiv_url"
          target="_blank"
          rel="noopener"
          class="inline-flex items-center gap-1 text-xs text-accent hover:underline mt-1.5"
        >
          View on arXiv ↗
        </a>
      </div>
    </Transition>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import type { Source } from '../../types'

defineProps<{
  index: number
  source?: Source
}>()

const show = ref(false)
</script>
