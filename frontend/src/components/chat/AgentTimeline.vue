<template>
  <div class="rounded-card border border-border bg-bg-secondary text-xs overflow-hidden">
    <!-- Header (always visible) -->
    <button
      type="button"
      class="w-full flex items-center justify-between px-3 py-2 text-left hover:bg-bg-hover transition-colors"
      @click="toggle"
    >
      <span class="flex items-center gap-2 min-w-0">
        <!-- status indicator -->
        <span v-if="running" class="relative flex h-2.5 w-2.5 shrink-0">
          <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-accent opacity-60"></span>
          <span class="relative inline-flex rounded-full h-2.5 w-2.5 bg-accent"></span>
        </span>
        <svg v-else class="w-3.5 h-3.5 text-ok shrink-0" viewBox="0 0 16 16" fill="none">
          <circle cx="8" cy="8" r="7" stroke="currentColor" stroke-width="1.5"/>
          <path d="M5 8.2l2 2 4-4.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>

        <!-- live label -->
        <span class="font-medium text-text-secondary truncate">
          {{ headerLabel }}
        </span>
        <span class="text-text-tertiary tabular-nums shrink-0">
          {{ progressLabel }} · {{ formatElapsed }}
        </span>
      </span>
      <svg
        class="w-3.5 h-3.5 text-text-tertiary shrink-0 transition-transform duration-200"
        :class="expanded ? 'rotate-180' : ''"
        viewBox="0 0 16 16" fill="none"
      >
        <path d="M4 6l4 4 4-4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
    </button>

    <!-- Timeline body -->
    <div v-if="expanded" class="border-t border-border px-3 py-2">
      <TransitionGroup name="tl" tag="ol" class="relative">
        <li
          v-for="item in items"
          :key="item.id"
          class="relative flex gap-2.5 py-1"
          :class="item.kind === 'step' ? 'pl-5' : ''"
        >
          <!-- connector line -->
          <span
            class="absolute top-0 bottom-0 w-px bg-border"
            :class="item.kind === 'step' ? 'left-[27.5px]' : 'left-[7.5px]'"
            aria-hidden="true"
          ></span>

          <!-- status icon -->
          <span class="relative z-10 mt-0.5 shrink-0 w-4 h-4 flex items-center justify-center bg-bg-secondary rounded-full">
            <svg v-if="item.status === 'done'" class="w-3.5 h-3.5 text-ok" viewBox="0 0 16 16" fill="none">
              <path d="M3.5 8.5l3 3 6-7" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            <span v-else-if="item.status === 'running'" class="w-3 h-3 rounded-full border-2 border-accent border-t-transparent animate-spin"></span>
            <svg v-else-if="item.status === 'warning'" class="w-3.5 h-3.5 text-warn" viewBox="0 0 16 16" fill="currentColor">
              <path d="M8 1.5l7 12.5H1L8 1.5zM7.3 6v4h1.4V6H7.3zm0 5.2v1.5h1.4v-1.5H7.3z"/>
            </svg>
            <svg v-else-if="item.status === 'failed'" class="w-3.5 h-3.5 text-bad" viewBox="0 0 16 16" fill="none">
              <path d="M4 4l8 8M12 4l-8 8" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
            </svg>
            <span v-else class="w-2 h-2 rounded-full border border-text-tertiary bg-bg-secondary"></span>
          </span>

          <!-- content -->
          <div class="flex-1 min-w-0">
            <div class="flex items-baseline justify-between gap-2">
              <button
                type="button"
                class="text-left truncate"
                :class="[
                  item.status === 'pending' ? 'text-text-tertiary' : 'text-text-primary',
                  hasDetail(item) ? 'hover:text-accent cursor-pointer' : 'cursor-default',
                ]"
                @click="hasDetail(item) && toggleDetail(item.id)"
              >
                <span class="font-medium">{{ item.title }}</span>
                <span v-if="hasDetail(item)" class="ml-1 text-text-tertiary">{{ openDetails.has(item.id) ? '▾' : '▸' }}</span>
              </button>
              <span v-if="item.durationMs" class="text-text-tertiary tabular-nums shrink-0">
                {{ formatDuration(item.durationMs) }}
              </span>
            </div>
            <p v-if="item.summary" class="text-text-secondary mt-0.5 leading-relaxed">
              {{ item.summary }}
            </p>
            <p v-else-if="item.status === 'pending' && item.reason" class="text-text-tertiary mt-0.5 leading-relaxed">
              {{ item.reason }}
            </p>
            <pre
              v-if="openDetails.has(item.id) && hasDetail(item)"
              class="mt-1.5 p-2 rounded-md bg-bg-primary border border-border text-[11px] leading-snug text-text-secondary overflow-x-auto max-h-48 overflow-y-auto"
            >{{ formatDetail(item.detail) }}</pre>
          </div>
        </li>
      </TransitionGroup>
      <p v-if="items.length === 0" class="text-text-tertiary py-1">正在初始化…</p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import type { TimelineItem } from '../../types'
import { currentActivity, doneCount } from '../../utils/timeline'

const props = defineProps<{
  items: TimelineItem[]
  elapsedMs: number
  running: boolean
  /** True once answer tokens started streaming (auto-collapses the timeline). */
  answering?: boolean
}>()

// --- expand/collapse with user-override memory -----------------------------
const expanded = ref(props.running)
const userToggled = ref(false)

function toggle() {
  expanded.value = !expanded.value
  userToggled.value = true
}

watch(() => props.running, (running) => {
  if (userToggled.value) return
  expanded.value = running && !props.answering
}, { immediate: true })

watch(() => props.answering, (answering) => {
  // The answer is streaming — collapse so the eye moves to the content.
  if (answering && !userToggled.value) expanded.value = false
})

// --- live elapsed timer -----------------------------------------------------
const displayElapsedMs = ref(props.elapsedMs || 0)
let timer: ReturnType<typeof setInterval> | null = null
let startedAt = Date.now() - displayElapsedMs.value

watch(() => props.running, (running) => {
  if (running) {
    startedAt = Date.now() - (props.elapsedMs || 0)
    if (!timer) timer = setInterval(() => {
      displayElapsedMs.value = Math.max(displayElapsedMs.value, Date.now() - startedAt)
    }, 100)
  } else {
    if (timer) { clearInterval(timer); timer = null }
    displayElapsedMs.value = props.elapsedMs || displayElapsedMs.value
  }
}, { immediate: true })

watch(() => props.elapsedMs, (ms) => {
  if (props.running) {
    displayElapsedMs.value = Math.max(displayElapsedMs.value, ms || 0)
    startedAt = Date.now() - displayElapsedMs.value
  } else if (ms) {
    displayElapsedMs.value = ms
  }
})

onBeforeUnmount(() => { if (timer) clearInterval(timer) })

// --- header content ----------------------------------------------------------
const headerLabel = computed(() => {
  if (!props.running) return '已完成'
  if (props.answering) return '正在生成回答'
  const active = currentActivity(props.items)
  return active ? `正在${active.title}` : '正在思考'
})

const progressLabel = computed(() => {
  const total = props.items.length
  if (total === 0) return ''
  return `${doneCount(props.items)}/${total} 步`
})

const formatElapsed = computed(() => {
  const s = (displayElapsedMs.value || 0) / 1000
  if (s < 60) return `${s.toFixed(1)}s`
  const m = Math.floor(s / 60)
  return `${m}m${(s - m * 60).toFixed(0)}s`
})

// --- detail drawer ----------------------------------------------------------
const openDetails = ref(new Set<string>())

function toggleDetail(id: string) {
  const next = new Set(openDetails.value)
  if (next.has(id)) next.delete(id)
  else next.add(id)
  openDetails.value = next
}

function hasDetail(item: TimelineItem): boolean {
  return !!item.detail && Object.keys(item.detail).length > 0
}

function formatDetail(detail?: Record<string, unknown>): string {
  try {
    return JSON.stringify(detail, null, 2)
  } catch {
    return String(detail)
  }
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`
  return `${(ms / 1000).toFixed(1)}s`
}
</script>

<style scoped>
.tl-enter-active {
  transition: opacity 0.25s ease, transform 0.25s ease;
}
.tl-enter-from {
  opacity: 0;
  transform: translateY(4px);
}
</style>
