<template>
  <span
    :class="['inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium flex-shrink-0', cls]"
  >相关性：{{ label }}</span>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { Relevance } from '../../types'

const props = defineProps<{ level: Relevance }>()

const label = computed(() => ({ high: '高', medium: '中', low: '低' }[props.level]))
const cls = computed(() => ({
  high:   'bg-ok/10 text-ok',
  medium: 'bg-warn/10 text-warn',
  low:    'bg-text-tertiary/10 text-text-tertiary',
}[props.level]))
</script>
