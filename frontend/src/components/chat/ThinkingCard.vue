<template>
  <div v-if="steps.length > 0" class="ml-0 mb-2">
    <div class="rounded-card border border-border bg-bg-secondary px-4 py-3">
      <div class="flex items-center justify-between mb-2 cursor-pointer" @click="expanded = !expanded">
        <span class="text-xs font-medium text-text-secondary">
          ⚡ Agent {{ isRunning ? '正在思考...' : '思考完成' }}
        </span>
        <span class="text-xs text-text-tertiary">{{ expanded ? '收起 ▴' : '展开 ▾' }}</span>
      </div>
      <div v-if="expanded" class="space-y-0.5">
        <StepIndicator v-for="step in steps" :key="step.index" :step="step" />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import type { ThinkingStep } from '../../types'
import StepIndicator from './StepIndicator.vue'

const props = defineProps<{ steps: ThinkingStep[] }>()
const expanded = ref(true)
const isRunning = computed(() => props.steps.some(s => s.status === 'running'))
</script>
