<template>
  <div class="flex-1 overflow-y-auto bg-bg-primary">
    <section class="max-w-5xl mx-auto px-6 py-6 space-y-5">
      <header class="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h2 class="text-lg font-semibold text-text-primary">我的论文库</h2>
          <p class="text-sm text-text-tertiary mt-1">浏览已入库论文，按关键词或类别筛选。</p>
        </div>
        <button
          type="button"
          class="px-3 py-2 rounded-md bg-accent text-white text-sm hover:opacity-90 disabled:opacity-50"
          :disabled="loading || overviewLoading"
          @click="refreshAll"
        >刷新</button>
      </header>

      <section class="rounded-card border border-border bg-bg-card">
        <button
          type="button"
          class="flex w-full items-start justify-between gap-4 px-4 py-4 text-left transition hover:bg-bg-hover"
          :aria-expanded="overviewExpanded"
          @click="overviewExpanded = !overviewExpanded"
        >
          <div class="min-w-0">
            <div class="flex flex-wrap items-center gap-2">
              <h3 class="text-sm font-semibold text-text-primary">语料库详情</h3>
              <span class="rounded-sm bg-bg-hover px-1.5 py-0.5 text-xs text-text-tertiary">
                {{ overviewBadge }}
              </span>
            </div>
            <p class="mt-1 text-sm leading-6 text-text-secondary">
              {{ overviewSummary }}
            </p>
          </div>
          <span class="mt-0.5 shrink-0 text-xs text-text-tertiary">
            {{ overviewExpanded ? '收起' : '展开' }}
          </span>
        </button>

        <div v-if="overviewExpanded" class="border-t border-border p-4">
          <CorpusOverviewCard
            :overview="overview"
            :loading="overviewLoading"
            :error="overviewError"
            variant="detailed"
            :show-header="false"
          />
        </div>
      </section>

      <div class="grid gap-3 sm:grid-cols-[1fr_180px_120px]">
        <input
          v-model="query"
          class="px-3 py-2 rounded-md border border-border bg-bg-card text-sm outline-none focus:border-accent"
          placeholder="搜索标题或摘要"
          @keyup.enter="load"
        />
        <input
          v-model="category"
          class="px-3 py-2 rounded-md border border-border bg-bg-card text-sm outline-none focus:border-accent"
          placeholder="类别，如 cs.CL"
          @keyup.enter="load"
        />
        <button
          type="button"
          class="px-3 py-2 rounded-md border border-border text-sm hover:bg-bg-hover"
          @click="load"
        >查询</button>
      </div>

      <div v-if="error" class="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
        {{ error }}
      </div>

      <div class="text-xs text-text-tertiary">共 {{ total }} 篇论文</div>

      <div class="space-y-2">
        <article
          v-for="paper in papers"
          :key="paper.paper_id"
          class="rounded-card border border-border bg-bg-card p-4"
        >
          <div class="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
            <div class="min-w-0">
              <h3 class="text-sm font-medium text-text-primary leading-snug">{{ paper.title }}</h3>
              <div class="mt-1 flex flex-wrap items-center gap-1.5 text-xs text-text-tertiary">
                <span>{{ paper.authors?.slice(0, 4).join(', ') || 'Unknown authors' }}</span>
                <span v-if="paper.year">· {{ paper.year }}</span>
                <span
                  v-if="paper.topic_bucket_label"
                  class="rounded-sm border border-border bg-bg-hover px-1.5 py-0.5 text-text-secondary"
                >{{ paper.topic_bucket_label }}</span>
                <span v-if="paper.primary_category">arXiv: {{ paper.primary_category }}</span>
              </div>
            </div>
            <a
              v-if="paper.arxiv_url"
              :href="paper.arxiv_url"
              target="_blank"
              rel="noopener"
              class="text-xs text-accent hover:underline flex-shrink-0"
            >arXiv</a>
          </div>
          <p v-if="paper.abstract" class="mt-2 text-sm text-text-secondary line-clamp-3">
            {{ paper.abstract }}
          </p>
          <p class="mt-2 text-xs text-text-tertiary">状态 {{ paper.ingest_status }} · {{ paper.num_chunks }} chunks</p>
        </article>
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { getCorpusOverview, listPapers } from '../api/papers'
import CorpusOverviewCard from '../components/chat/CorpusOverviewCard.vue'
import type { CorpusOverviewResponse, PaperSummary } from '../types'

const papers = ref<PaperSummary[]>([])
const total = ref(0)
const query = ref('')
const category = ref('')
const loading = ref(false)
const error = ref('')
const overview = ref<CorpusOverviewResponse | null>(null)
const overviewLoading = ref(false)
const overviewError = ref('')
const overviewExpanded = ref(true)

const overviewBadge = computed(() => {
  if (overviewLoading.value) return '加载中'
  if (overviewError.value) return '暂不可用'
  if (!overview.value) return '未加载'
  return `${overview.value.total_papers} 篇论文`
})

const overviewSummary = computed(() => {
  if (overviewLoading.value) return '正在读取当前论文库的主题分布。'
  if (overviewError.value) return '语料库详情暂时不可用，不影响论文列表浏览。'
  if (!overview.value || overview.value.total_papers === 0) return '还没有可检索的论文。'
  const years = overview.value.year_min && overview.value.year_max
    ? `${overview.value.year_min}-${overview.value.year_max}`
    : '年份未知'
  const topTopics = overview.value.topic_buckets
    .slice(0, 3)
    .map(bucket => `${bucket.label} ${bucket.paper_count}`)
    .join(' · ')
  return `${overview.value.total_papers} 篇论文，${overview.value.total_chunks.toLocaleString()} 个片段，覆盖 ${years}；主要方向：${topTopics}。`
})

onMounted(() => {
  void refreshAll()
})

async function refreshAll() {
  await Promise.all([load(), loadOverview()])
}

async function load() {
  loading.value = true
  error.value = ''
  try {
    const data = await listPapers({
      q: query.value || undefined,
      category: category.value || undefined,
      limit: 50,
    })
    papers.value = data.items
    total.value = data.total
  } catch {
    error.value = '论文列表加载失败，请稍后重试。'
  } finally {
    loading.value = false
  }
}

async function loadOverview() {
  overviewLoading.value = true
  overviewError.value = ''
  try {
    overview.value = await getCorpusOverview()
  } catch {
    overviewError.value = '语料库详情加载失败，请稍后重试。'
  } finally {
    overviewLoading.value = false
  }
}
</script>
