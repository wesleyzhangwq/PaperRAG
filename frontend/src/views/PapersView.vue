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
          :disabled="loading"
          @click="load"
        >刷新</button>
      </header>

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
              <p class="mt-1 text-xs text-text-tertiary">
                {{ paper.authors?.slice(0, 4).join(', ') || 'Unknown authors' }}
                <span v-if="paper.year"> · {{ paper.year }}</span>
                <span v-if="paper.primary_category"> · {{ paper.primary_category }}</span>
              </p>
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
import { onMounted, ref } from 'vue'
import { listPapers } from '../api/papers'
import type { PaperSummary } from '../types'

const papers = ref<PaperSummary[]>([])
const total = ref(0)
const query = ref('')
const category = ref('')
const loading = ref(false)
const error = ref('')

onMounted(load)

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
</script>
