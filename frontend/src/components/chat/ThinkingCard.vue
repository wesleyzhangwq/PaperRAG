<template>
  <div class="rounded-card border border-border bg-bg-secondary text-xs">
    <!-- Header: status + timer + expand toggle -->
    <button
      type="button"
      class="w-full flex items-center justify-between px-3 py-2 text-left"
      @click="expanded = !expanded"
    >
      <span class="flex items-center gap-2">
        <span
          v-if="running"
          class="inline-block w-2 h-2 rounded-full bg-accent animate-pulse"
        ></span>
        <span v-else class="inline-block w-2 h-2 rounded-full bg-ok"></span>
        <span class="font-medium text-text-secondary">
          {{ running ? '正在思考' : '执行步骤' }}
        </span>
        <span class="text-text-tertiary tabular-nums">
          · {{ formatElapsed }} · {{ doneCount }}/{{ steps.length || '?' }} 步
        </span>
      </span>
      <span class="text-text-tertiary">{{ expanded ? '收起 ▴' : '展开 ▾' }}</span>
    </button>

    <!-- Body -->
    <div v-if="expanded" class="border-t border-border px-3 py-2 space-y-1">
      <StepIndicator
        v-for="step in steps"
        :key="step.index"
        :step="step"
        :call="callForIndex(step.index)"
        :result="resultForIndex(step.index)"
      />
      <p v-if="steps.length === 0" class="text-text-tertiary py-1">
        正在等待 agent 生成计划…
      </p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import type { ThinkingStep, ToolCallEvent, ToolResultEvent } from '../../types'
import StepIndicator from './StepIndicator.vue'

const props = defineProps<{
  steps: ThinkingStep[]
  toolCalls: ToolCallEvent[]
  toolResults: ToolResultEvent[]
  elapsedMs: number
  running: boolean
}>()

const expanded = ref(props.running)
const displayElapsedMs = ref(props.elapsedMs || 0)
let timer: ReturnType<typeof setInterval> | null = null
let localStartedAt = Date.now() - displayElapsedMs.value

watch(
  () => props.running,
  (running) => {
    if (running) {
      expanded.value = true
      localStartedAt = Date.now() - (props.elapsedMs || displayElapsedMs.value || 0)
      startTimer()
    } else {
      displayElapsedMs.value = props.elapsedMs || displayElapsedMs.value
      expanded.value = false
      stopTimer()
    }
  },
  { immediate: true },
)

watch(
  () => props.elapsedMs,
  (elapsed) => {
    if (props.running) {
      displayElapsedMs.value = Math.max(displayElapsedMs.value, elapsed || 0)
      localStartedAt = Date.now() - displayElapsedMs.value
    } else {
      displayElapsedMs.value = elapsed || displayElapsedMs.value
    }
  },
)

onBeforeUnmount(stopTimer)

const doneCount = computed(
  () => props.steps.filter(s => s.status === 'done' || s.status === 'failed').length
)

const completedStepMs = computed(() =>
  props.steps.reduce((sum, step) => sum + (step.durationMs || 0), 0)
)

const formatElapsed = computed(() => {
  const ms = props.running ? displayElapsedMs.value : completedStepMs.value
  const s = (ms || 0) / 1000
  if (s < 60) return `${s.toFixed(1)}s`
  const m = Math.floor(s / 60)
  const r = (s - m * 60).toFixed(0)
  return `${m}m${r}s`
})

function startTimer() {
  if (timer) return
  timer = setInterval(() => {
    displayElapsedMs.value = Math.max(displayElapsedMs.value, Date.now() - localStartedAt)
  }, 100)
}

function stopTimer() {
  if (!timer) return
  clearInterval(timer)
  timer = null
}

function callForIndex(i: number): ToolCallEvent | undefined {
  return props.toolCalls.find(c => c.index === i)
}
function resultForIndex(i: number): ToolResultEvent | undefined {
  return props.toolResults.find(r => r.index === i)
}
</script>
