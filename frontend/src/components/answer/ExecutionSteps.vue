<template>
  <div class="rounded-card border border-border bg-bg-secondary">
    <button
      type="button"
      class="w-full px-3 py-2 flex items-center justify-between text-left"
      @click="open = !open"
    >
      <span class="flex items-center gap-2 text-xs font-medium text-text-secondary">
        <span>⚙️ 执行步骤</span>
        <span class="text-text-tertiary font-normal">
          · {{ doneCount }}/{{ steps.length }} 完成{{ warnCount ? ` · ${warnCount} 警告` : '' }}
        </span>
      </span>
      <span class="text-xs text-text-tertiary">{{ open ? '收起 ▴' : '展开 ▾' }}</span>
    </button>
    <ol v-if="open" class="border-t border-border px-3 py-2 space-y-1.5">
      <li
        v-for="(s, i) in steps"
        :key="s.index"
        class="flex items-start gap-2 text-sm"
      >
        <span class="flex-shrink-0 w-5 text-xs text-text-tertiary font-mono mt-0.5">{{ i + 1 }}.</span>
        <span class="flex-shrink-0 mt-1">
          <span v-if="s.status === 'completed'" class="text-ok">●</span>
          <span v-else-if="s.status === 'warning'" class="text-warn">▲</span>
          <span v-else-if="s.status === 'error'" class="text-bad">✕</span>
          <span v-else class="text-text-tertiary animate-pulse">○</span>
        </span>
        <div class="flex-1 min-w-0">
          <div class="flex items-baseline gap-2">
            <span class="font-medium text-text-primary">{{ s.name }}</span>
            <span class="text-xs text-text-tertiary tabular-nums">
              {{ formatDuration(s.duration_ms) }}
            </span>
          </div>
          <p
            :class="[
              'text-xs leading-relaxed',
              s.status === 'warning' ? 'text-warn' :
              s.status === 'error' ? 'text-bad' : 'text-text-secondary'
            ]"
          >{{ s.user_message }}</p>
        </div>
      </li>
    </ol>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import type { PresentationStep } from '../../types'

const props = defineProps<{ steps: PresentationStep[] }>()
const open = ref(true)

const doneCount = computed(() =>
  props.steps.filter(s => s.status === 'completed').length
)
const warnCount = computed(() =>
  props.steps.filter(s => s.status === 'warning' || s.status === 'error').length
)

function formatDuration(ms: number): string {
  if (!ms) return ''
  if (ms < 1000) return `${Math.round(ms)}ms`
  return `${(ms / 1000).toFixed(1)}s`
}
</script>
