<template>
  <span
    :class="['inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium border', cls]"
    :title="reason"
  >
    <span class="w-1.5 h-1.5 rounded-full" :class="dotCls"></span>
    可信度：{{ label }}
  </span>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { Confidence } from '../../types'

const props = defineProps<{ level: Confidence; reason?: string }>()

const label = computed(() => ({ high: '高', medium: '中', low: '低' }[props.level]))
const cls = computed(() => ({
  high:   'bg-ok/10 text-ok border-ok/30',
  medium: 'bg-warn/10 text-warn border-warn/30',
  low:    'bg-bad/10 text-bad border-bad/30',
}[props.level]))
const dotCls = computed(() => ({
  high: 'bg-ok',
  medium: 'bg-warn',
  low: 'bg-bad',
}[props.level]))
</script>
