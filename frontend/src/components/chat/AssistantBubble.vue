<template>
  <div class="flex justify-start">
    <div class="max-w-[80%] px-4 py-3 rounded-card bg-bg-card shadow-card text-sm text-text-primary prose prose-sm" v-html="rendered"></div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { renderMarkdown } from '../../utils/markdown'
import type { Source } from '../../types'

const props = defineProps<{ content: string; sources?: Source[] }>()

const rendered = computed(() => renderMarkdown(props.content, props.sources))
</script>
