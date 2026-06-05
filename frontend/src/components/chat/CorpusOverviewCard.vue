<template>
  <section class="w-full max-w-3xl mx-auto text-left">
    <div class="mb-5">
      <p class="text-xs font-medium uppercase text-text-tertiary">Corpus Overview</p>
      <h2 class="mt-2 text-2xl font-semibold text-text-primary">这套 RAG 现在收录了什么？</h2>
      <p class="mt-2 max-w-2xl text-sm leading-6 text-text-secondary">
        先快速看一眼本地论文库覆盖范围，再决定从哪个方向提问。
      </p>
    </div>

    <div v-if="loading" class="rounded-card border border-border bg-bg-card p-5">
      <div class="h-5 w-40 rounded bg-bg-hover"></div>
      <div class="mt-4 grid grid-cols-2 gap-px overflow-hidden rounded-md border border-border bg-border md:grid-cols-4">
        <div v-for="index in 4" :key="index" class="bg-bg-primary p-3">
          <div class="h-3 w-16 rounded bg-bg-hover"></div>
          <div class="mt-3 h-5 w-20 rounded bg-bg-hover"></div>
        </div>
      </div>
    </div>

    <div v-else-if="error" class="rounded-card border border-border bg-bg-card p-5 text-sm text-text-secondary">
      语料概览暂时不可用。你仍然可以直接提问或进入论文库查看。
    </div>

    <div v-else-if="!overview || overview.total_papers === 0" class="rounded-card border border-border bg-bg-card p-5">
      <h3 class="text-base font-semibold text-text-primary">还没有可检索的论文</h3>
      <p class="mt-2 text-sm leading-6 text-text-secondary">
        上传或 ingest 论文后，这里会显示主题分布、代表论文和推荐问题。
      </p>
      <div class="mt-4 flex flex-wrap gap-2">
        <button
          v-for="question in fallbackQuestions"
          :key="question"
          type="button"
          class="rounded-md border border-border px-3 py-2 text-sm text-text-secondary hover:border-accent hover:text-accent"
          @click="$emit('ask', question)"
        >
          {{ question }}
        </button>
      </div>
    </div>

    <div v-else class="rounded-card border border-border bg-bg-card">
      <div class="grid grid-cols-2 gap-px overflow-hidden rounded-t-card bg-border md:grid-cols-4">
        <div class="bg-bg-primary p-3">
          <p class="text-xs text-text-tertiary">Papers</p>
          <p class="mt-1 text-xl font-semibold text-text-primary">{{ overview.total_papers }}</p>
        </div>
        <div class="bg-bg-primary p-3">
          <p class="text-xs text-text-tertiary">Chunks</p>
          <p class="mt-1 text-xl font-semibold text-text-primary">{{ formattedChunks }}</p>
        </div>
        <div class="bg-bg-primary p-3">
          <p class="text-xs text-text-tertiary">Years</p>
          <p class="mt-1 text-xl font-semibold text-text-primary">{{ yearRange }}</p>
        </div>
        <div class="bg-bg-primary p-3">
          <p class="text-xs text-text-tertiary">Topics</p>
          <p class="mt-1 text-xl font-semibold text-text-primary">{{ overview.topic_buckets.length }}</p>
        </div>
      </div>

      <div class="divide-y divide-border">
        <article v-for="bucket in overview.topic_buckets" :key="bucket.key" class="p-4">
          <div class="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
            <div class="min-w-0">
              <div class="flex items-center gap-2">
                <h3 class="text-sm font-semibold text-text-primary">{{ bucket.label }}</h3>
                <span class="rounded-sm bg-bg-hover px-1.5 py-0.5 text-xs text-text-tertiary">
                  {{ bucket.paper_count }} papers
                </span>
              </div>
              <p class="mt-1 text-xs leading-5 text-text-secondary">{{ bucket.description }}</p>
            </div>
            <div class="h-1.5 w-full rounded-full bg-bg-hover md:w-28">
              <div class="h-1.5 rounded-full bg-accent" :style="{ width: topicWidth(bucket.paper_count) }"></div>
            </div>
          </div>

          <ul v-if="bucket.representative_papers.length" class="mt-3 space-y-1.5">
            <li
              v-for="paper in bucket.representative_papers"
              :key="paper.paper_id"
              class="text-xs leading-5 text-text-secondary"
            >
              <a
                :href="paper.arxiv_url"
                target="_blank"
                rel="noopener"
                class="hover:text-accent hover:underline"
              >
                {{ paper.title }}
              </a>
              <span class="text-text-tertiary">
                <span v-if="paper.year"> · {{ paper.year }}</span>
                <span v-if="paper.primary_category"> · {{ paper.primary_category }}</span>
              </span>
            </li>
          </ul>
        </article>
      </div>

      <div v-if="overview.suggested_questions.length" class="border-t border-border p-4">
        <p class="text-xs text-text-tertiary">可以从这些问题开始</p>
        <div class="mt-3 flex flex-wrap gap-2">
          <button
            v-for="question in overview.suggested_questions"
            :key="question"
            type="button"
            class="rounded-md border border-border px-3 py-2 text-sm text-text-secondary transition hover:border-accent hover:text-accent"
            @click="$emit('ask', question)"
          >
            {{ question }}
          </button>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { CorpusOverviewResponse } from '../../types'

const props = defineProps<{
  overview: CorpusOverviewResponse | null
  loading: boolean
  error: string
}>()

defineEmits<{ ask: [question: string] }>()

const fallbackQuestions = ['如何上传第一篇论文？']

const formattedChunks = computed(() => {
  const value = props.overview?.total_chunks || 0
  return value.toLocaleString()
})

const yearRange = computed(() => {
  if (!props.overview?.year_min || !props.overview?.year_max) return 'n/a'
  if (props.overview.year_min === props.overview.year_max) return String(props.overview.year_min)
  return `${props.overview.year_min}-${props.overview.year_max}`
})

const maxPaperCount = computed(() => {
  const counts = props.overview?.topic_buckets.map(bucket => bucket.paper_count) || []
  return Math.max(...counts, 1)
})

function topicWidth(count: number): string {
  const pct = Math.max(12, Math.round((count / maxPaperCount.value) * 100))
  return `${pct}%`
}
</script>
