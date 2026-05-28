<template>
  <div class="flex justify-start">
    <div class="max-w-full w-full">
      <div
        :class="[
          'px-4 py-3 rounded-card bg-bg-card shadow-card text-sm text-text-primary prose prose-sm max-w-none',
          streaming && !content ? 'min-h-[2.2rem] flex items-center text-text-tertiary italic' : '',
          streaming ? 'cursor-blink' : '',
        ]"
        v-html="rendered || (streaming ? '正在生成回答…' : '')"
        @mouseenter="handleHover"
        @mouseleave="hidePopover"
        ref="contentRef"
      ></div>

      <!-- Citation source list -->
      <div v-if="citedSources.length > 0" class="mt-2 px-1">
        <div class="flex flex-wrap gap-1.5">
          <a
            v-for="(src, i) in citedSources"
            :key="src.paper_id"
            :href="src.arxiv_url || '#'"
            target="_blank"
            rel="noopener"
            class="inline-flex items-center gap-1 px-2 py-1 rounded-md text-xs bg-bg-secondary hover:bg-bg-hover text-text-secondary hover:text-accent transition border border-border"
          >
            <span class="font-medium text-accent">[{{ i + 1 }}]</span>
            <span class="truncate max-w-[200px]">{{ src.title }}</span>
            <span v-if="src.year" class="text-text-tertiary">{{ src.year }}</span>
          </a>
        </div>
      </div>

      <!-- Floating popover -->
      <Transition
        enter-active-class="transition duration-150 ease-out"
        enter-from-class="opacity-0 scale-95"
        enter-to-class="opacity-100 scale-100"
        leave-active-class="transition duration-100 ease-in"
        leave-from-class="opacity-100 scale-100"
        leave-to-class="opacity-0 scale-95"
      >
        <div
          v-if="popoverSource && popoverVisible"
          class="fixed z-50 w-72 bg-bg-card border border-border rounded-lg shadow-lg p-3"
          :style="{ top: popoverTop + 'px', left: popoverLeft + 'px' }"
          @mouseenter="popoverVisible = true"
          @mouseleave="hidePopover"
        >
          <p class="text-sm font-medium text-text-primary leading-tight mb-1">{{ popoverSource.title }}</p>
          <p class="text-xs text-text-secondary mb-1">
            {{ popoverSource.authors?.join(', ') || 'Unknown' }}
            <span v-if="popoverSource.year"> · {{ popoverSource.year }}</span>
            <span v-if="popoverSource.primary_category"> · {{ popoverSource.primary_category }}</span>
          </p>
          <p v-if="popoverSource.score" class="text-xs text-text-tertiary mb-1">
            Score: {{ popoverSource.score.toFixed(2) }}
            <span v-if="popoverSource.page_num"> · Page {{ popoverSource.page_num }}</span>
          </p>
          <p v-if="popoverSource.snippet" class="text-xs text-text-secondary italic border-t border-border pt-1.5 mt-1.5 line-clamp-3">
            "{{ popoverSource.snippet }}"
          </p>
          <a
            v-if="popoverSource.arxiv_url"
            :href="popoverSource.arxiv_url"
            target="_blank"
            rel="noopener"
            class="inline-flex items-center gap-1 text-xs text-accent hover:underline mt-1.5"
          >
            View on arXiv ↗
          </a>
        </div>
      </Transition>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import { renderMarkdown, extractCitedIds } from '../../utils/markdown'
import type { Source } from '../../types'

const props = defineProps<{ content: string; sources?: Source[]; streaming?: boolean }>()

const contentRef = ref<HTMLElement | null>(null)
const popoverVisible = ref(false)
const popoverSource = ref<Source | null>(null)
const popoverTop = ref(0)
const popoverLeft = ref(0)

const rendered = computed(() => renderMarkdown(props.content, props.sources))

const citedSources = computed(() => {
  if (!props.sources?.length) return []
  const ids = extractCitedIds(props.content)
  return ids
    .map(id => props.sources!.find(s => s.paper_id === id))
    .filter((s): s is Source => !!s)
})

let hideTimeout: ReturnType<typeof setTimeout> | null = null

function handleHover(e: MouseEvent) {
  const target = e.target as HTMLElement
  const pill = target.closest('[data-paper-id]') as HTMLElement | null
  if (!pill) return

  if (hideTimeout) clearTimeout(hideTimeout)

  const paperId = pill.dataset.paperId
  const source = props.sources?.find(s => s.paper_id === paperId)
  if (!source) return

  const rect = pill.getBoundingClientRect()
  popoverTop.value = rect.bottom + 8
  popoverLeft.value = Math.max(8, rect.left - 100)
  popoverSource.value = source
  popoverVisible.value = true
}

function hidePopover() {
  hideTimeout = setTimeout(() => {
    popoverVisible.value = false
    popoverSource.value = null
  }, 150)
}
</script>
