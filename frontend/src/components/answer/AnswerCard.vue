<template>
  <article class="space-y-3">
    <!-- ① 回答 -->
    <div class="rounded-card bg-bg-card border border-border shadow-card p-4">
      <header class="flex items-center justify-between mb-2">
        <span class="text-xs font-medium text-text-tertiary uppercase tracking-wide">回答</span>
        <ConfidenceBadge
          v-if="presentation"
          :level="presentation.confidence"
          :reason="presentation.confidence_reason"
        />
      </header>
      <div
        :class="[
          'prose prose-sm max-w-none text-text-primary text-sm',
          streaming ? 'cursor-blink' : '',
        ]"
        v-html="rendered || (streaming ? '<em class=\'text-text-tertiary\'>正在生成回答…</em>' : '')"
        @mouseenter="onHover"
        @mouseleave="hidePopover"
      ></div>
      <p
        v-if="presentation?.confidence_reason"
        class="mt-2 text-xs text-text-tertiary border-t border-border pt-2"
      >
        <span class="font-medium">可信度说明：</span>{{ presentation.confidence_reason }}
      </p>
    </div>

    <!-- ② 参考论文 -->
    <section v-if="cards.length > 0">
      <h3 class="text-xs font-medium text-text-tertiary uppercase tracking-wide mb-2">
        参考论文 · {{ cards.length }} 篇
      </h3>
      <div class="space-y-2">
        <SourceCardComp
          v-for="(c, i) in cards"
          :key="c.paper_id"
          :source="c"
          :index="i + 1"
        />
      </div>
    </section>

    <!-- ③ 检索概况 -->
    <RetrievalSummaryCard v-if="presentation?.retrieval_summary" :summary="presentation.retrieval_summary" />

    <!-- ④ 执行步骤 -->
    <ExecutionSteps v-if="presentation?.steps && presentation.steps.length > 0" :steps="presentation.steps" />

    <!-- ⑤ 调试详情（默认折叠） -->
    <DebugPanel
      v-if="presentation?.steps && presentation.steps.length > 0"
      :steps="presentation.steps"
      :raw-sources="sources"
    />

    <!-- Citation popover (carried over from AssistantBubble) -->
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
        class="fixed z-50 w-72 bg-bg-card border border-border rounded-lg shadow-popover p-3"
        :style="{ top: popoverTop + 'px', left: popoverLeft + 'px' }"
        @mouseenter="popoverVisible = true"
        @mouseleave="hidePopover"
      >
        <p class="text-sm font-medium text-text-primary leading-tight mb-1">{{ popoverSource.title }}</p>
        <p class="text-xs text-text-secondary mb-1">
          {{ popoverSource.authors?.join(', ') || 'Unknown' }}
          <span v-if="popoverSource.year"> · {{ popoverSource.year }}</span>
        </p>
        <a
          v-if="popoverSource.arxiv_url"
          :href="popoverSource.arxiv_url"
          target="_blank"
          rel="noopener"
          class="inline-flex items-center gap-1 text-xs text-accent hover:underline mt-1.5"
        >View on arXiv ↗</a>
      </div>
    </Transition>
  </article>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import { renderMarkdown } from '../../utils/markdown'
import type { Presentation, Source } from '../../types'
import ConfidenceBadge from './ConfidenceBadge.vue'
import SourceCardComp from './SourceCard.vue'
import RetrievalSummaryCard from './RetrievalSummaryCard.vue'
import ExecutionSteps from './ExecutionSteps.vue'
import DebugPanel from './DebugPanel.vue'

const props = defineProps<{
  content: string
  sources?: Source[]
  presentation?: Presentation | null
  streaming?: boolean
}>()

const rendered = computed(() => renderMarkdown(props.content, props.sources))
const cards = computed(() => props.presentation?.sources || [])

// Hover popover for inline citations
const popoverVisible = ref(false)
const popoverSource = ref<Source | null>(null)
const popoverTop = ref(0)
const popoverLeft = ref(0)
let hideTimeout: ReturnType<typeof setTimeout> | null = null

function onHover(e: MouseEvent) {
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
