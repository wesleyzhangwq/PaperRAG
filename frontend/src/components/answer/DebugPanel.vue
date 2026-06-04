<template>
  <details class="rounded-card border border-dashed border-border bg-bg-secondary text-xs">
    <summary class="px-3 py-2 cursor-pointer text-text-tertiary select-none hover:text-text-secondary list-none flex items-center gap-2">
      <span>🛠 查看调试详情</span>
      <button
        type="button"
        class="ml-auto rounded border border-border px-2 py-0.5 text-[11px] text-text-secondary hover:bg-bg-hover"
        @click.stop.prevent="copyDebugDetails"
      >
        {{ copied ? '已复制' : '复制' }}
      </button>
      <span class="opacity-60">面向开发者</span>
    </summary>
    <div class="border-t border-border px-3 py-2 space-y-3 font-mono text-[11px] text-text-secondary">
      <div v-for="(s, i) in steps" :key="s.index">
        <div class="text-text-primary font-semibold">
          [{{ i + 1 }}] {{ s.name }} <span class="text-text-tertiary font-normal">— {{ s.debug.tool }}</span>
        </div>
        <div class="ml-2 mt-0.5 space-y-1">
          <div v-if="hasParams(s.debug.params)">
            <span class="text-text-tertiary">params:</span>
            <pre class="bg-bg-card rounded px-2 py-1 mt-0.5 overflow-x-auto whitespace-pre-wrap break-words">{{ format(s.debug.params) }}</pre>
          </div>
          <div v-if="s.debug.reason">
            <span class="text-text-tertiary">reason:</span>
            <span class="ml-1">{{ s.debug.reason }}</span>
          </div>
          <div v-if="s.debug.raw_summary">
            <span class="text-text-tertiary">raw_summary:</span>
            <span class="ml-1">{{ s.debug.raw_summary }}</span>
          </div>
          <div v-if="hasParams(s.debug.extra)">
            <span class="text-text-tertiary">extra:</span>
            <pre class="bg-bg-card rounded px-2 py-1 mt-0.5 overflow-x-auto whitespace-pre-wrap break-words">{{ format(s.debug.extra) }}</pre>
          </div>
        </div>
      </div>
      <div v-if="rawSources && rawSources.length > 0">
        <div class="text-text-primary font-semibold">原始 sources</div>
        <pre class="bg-bg-card rounded px-2 py-1 mt-0.5 overflow-x-auto whitespace-pre-wrap break-words">{{ format(rawSources) }}</pre>
      </div>
    </div>
  </details>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import type { PresentationStep, Source } from '../../types'

const props = defineProps<{ steps: PresentationStep[]; rawSources?: Source[] }>()
const copied = ref(false)

const debugText = computed(() => {
  const lines: string[] = []
  props.steps.forEach((s, i) => {
    lines.push(`[${i + 1}] ${s.name} — ${s.debug.tool}`)
    if (hasParams(s.debug.params)) lines.push(`params:\n${format(s.debug.params)}`)
    if (s.debug.reason) lines.push(`reason:${s.debug.reason}`)
    if (s.debug.raw_summary) lines.push(`raw_summary:${s.debug.raw_summary}`)
    if (hasParams(s.debug.extra)) lines.push(`extra:\n${format(s.debug.extra)}`)
  })
  if (props.rawSources && props.rawSources.length > 0) {
    lines.push(`原始 sources\n${format(props.rawSources)}`)
  }
  return lines.join('\n')
})

function format(v: unknown): string {
  try { return JSON.stringify(v, null, 2) } catch { return String(v) }
}
function hasParams(p: unknown): boolean {
  return !!p && typeof p === 'object' && Object.keys(p as object).length > 0
}

async function copyDebugDetails() {
  try {
    await navigator.clipboard.writeText(debugText.value)
  } catch {
    const textarea = document.createElement('textarea')
    textarea.value = debugText.value
    textarea.style.position = 'fixed'
    textarea.style.opacity = '0'
    document.body.appendChild(textarea)
    textarea.select()
    document.execCommand('copy')
    document.body.removeChild(textarea)
  }
  copied.value = true
  window.setTimeout(() => {
    copied.value = false
  }, 1400)
}
</script>
