<template>
  <div class="flex items-start gap-2 py-1.5">
    <span class="mt-0.5 text-xs">
      <span v-if="step.status === 'running'" class="animate-spin inline-block">◐</span>
      <span v-else-if="step.status === 'done'" class="text-green-600">●</span>
      <span v-else-if="step.status === 'failed'" class="text-red-500">●</span>
      <span v-else class="text-text-tertiary">○</span>
    </span>
    <div class="flex-1 min-w-0">
      <div class="flex items-center gap-2">
        <span class="text-xs font-medium text-text-primary">{{ actionLabel }}</span>
        <span v-if="step.durationMs" class="text-xs text-text-tertiary">{{ step.durationMs }}ms</span>
      </div>
      <p v-if="step.outputSummary" class="text-xs text-text-secondary mt-0.5 truncate">{{ step.outputSummary }}</p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { ThinkingStep } from '../../types'

const props = defineProps<{ step: ThinkingStep }>()

const actionLabels: Record<string, string> = {
  intent_analysis: '意图分析',
  planning: '生成计划',
  query_rewrite: '查询改写',
  retrieve_local: '本地检索',
  retrieve_arxiv: 'arXiv 搜索',
  search_web: '网页搜索',
  evaluate_docs: '充分性评估',
  reasoning_synthesis: '推理生成',
  self_reflection: '自我验证',
  re_planning: '重新规划',
}

const actionLabel = computed(() => actionLabels[props.step.action] || props.step.action)
</script>
