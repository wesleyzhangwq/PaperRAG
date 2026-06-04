<template>
  <div class="text-xs">
    <button
      type="button"
      class="w-full flex items-start gap-2 py-1 text-left hover:bg-bg-hover rounded px-1 transition"
      @click="open = !open"
    >
      <span class="mt-0.5 flex-shrink-0 w-3">
        <span v-if="step.status === 'running'" class="text-accent animate-pulse">◐</span>
        <span v-else-if="step.status === 'done'" class="text-ok">●</span>
        <span v-else-if="step.status === 'failed'" class="text-bad">✕</span>
        <span v-else class="text-text-tertiary">○</span>
      </span>
      <div class="flex-1 min-w-0">
        <div class="flex items-center gap-2">
          <span class="font-medium text-text-primary">{{ actionLabel }}</span>
          <code v-if="actionLabel !== step.action" class="text-text-tertiary font-mono">
            {{ step.action }}
          </code>
          <span v-if="step.durationMs" class="text-text-tertiary tabular-nums">
            {{ formatStepDuration(step.durationMs) }}
          </span>
        </div>
        <p
          v-if="step.outputSummary"
          class="text-text-secondary mt-0.5 truncate"
        >{{ step.outputSummary }}</p>
        <p
          v-else-if="step.reason && step.status === 'pending'"
          class="text-text-tertiary mt-0.5 truncate"
        >{{ step.reason }}</p>
      </div>
      <span class="text-text-tertiary flex-shrink-0">{{ open ? '▴' : '▾' }}</span>
    </button>

    <!-- Expanded detail: params + result -->
    <div
      v-if="open"
      class="ml-5 mt-1 mb-2 space-y-2 text-[11px] font-mono text-text-secondary"
    >
      <div v-if="step.reason" class="pl-2 border-l-2 border-border">
        <span class="text-text-tertiary">reason: </span>{{ step.reason }}
      </div>

      <div v-if="call?.params && Object.keys(call.params).length > 0">
        <div class="text-text-tertiary mb-0.5">params:</div>
        <pre class="bg-bg-card rounded px-2 py-1 overflow-x-auto whitespace-pre-wrap break-words">{{ formatParams(call.params) }}</pre>
      </div>

      <div v-if="result?.detail && hasDetail">
        <div class="text-text-tertiary mb-0.5">result:</div>
        <pre class="bg-bg-card rounded px-2 py-1 overflow-x-auto whitespace-pre-wrap break-words">{{ formatDetail(result.detail) }}</pre>
      </div>

      <div
        v-if="!call && !result && step.status !== 'pending'"
        class="text-text-tertiary italic"
      >
        没有更多细节
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import type { ThinkingStep, ToolCallEvent, ToolResultEvent } from '../../types'
import { formatStepDuration } from '../../utils/duration'

const props = defineProps<{
  step: ThinkingStep
  call?: ToolCallEvent
  result?: ToolResultEvent
}>()

const open = ref(false)

const actionLabels: Record<string, string> = {
  intent_analysis: '意图分析',
  planning: '生成计划',
  query_rewrite: '查询改写',
  retrieve_local: '本地检索',
  retrieve_arxiv: 'arXiv 搜索',
  search_web: '网页搜索',
  evaluate_docs: '充分性评估',
  get_paper_detail: '论文详情',
  get_paper_chunks: '论文片段',
  reasoning_synthesis: '推理生成',
  self_reflection: '自我验证',
  re_planning: '重新规划',
}

const actionLabel = computed(() => actionLabels[props.step.action] || props.step.action)
const hasDetail = computed(() =>
  props.result?.detail && Object.keys(props.result.detail).length > 0
)

function formatParams(p: Record<string, unknown>): string {
  try { return JSON.stringify(p, null, 2) } catch { return String(p) }
}
function formatDetail(d: Record<string, unknown>): string {
  try { return JSON.stringify(d, null, 2) } catch { return String(d) }
}
</script>
