<script setup lang="ts">
import { onMounted } from 'vue'
import { NButton, NCard, NInput, NSelect, NTag, NSpace, NScrollbar, NEmpty } from 'naive-ui'
import { usePapersStore } from '../stores/papers'

const store = usePapersStore()

import type { SelectOption } from 'naive-ui'

const categoryOptions: SelectOption[] = [
  { label: '全部', value: '__all__' },
  { label: 'cs.AI', value: 'cs.AI' },
  { label: 'cs.CL', value: 'cs.CL' },
  { label: 'cs.LG', value: 'cs.LG' },
  { label: '用户上传', value: 'user.upload' },
]

const yearOptions: SelectOption[] = [
  { label: '不限', value: 0 },
  ...Array.from({ length: 4 }, (_, i) => 2023 + i).map((y) => ({ label: `${y}+`, value: y })),
]

function applyFilters() {
  // Treat sentinel values as null for API
  if ((store.filters.category as any) === '__all__') store.filters.category = null
  if (store.filters.year_min === 0) store.filters.year_min = null
  store.load()
}

function openArxiv(id: string) {
  window.open(`https://arxiv.org/abs/${id}`, '_blank')
}

onMounted(() => store.load())
</script>

<template>
  <div class="paper-list">
    <div class="filters">
      <NSpace vertical :size="8">
        <NInput
          v-model:value="store.filters.q"
          placeholder="搜索标题 / 摘要"
          clearable
          size="small"
          @change="applyFilters"
        />
        <NSelect
          v-model:value="store.filters.category"
          :options="categoryOptions"
          size="small"
          placeholder="分类"
          @update:value="applyFilters"
        />
        <NSelect
          v-model:value="store.filters.year_min"
          :options="yearOptions"
          size="small"
          placeholder="年份"
          @update:value="applyFilters"
        />
        <NButton size="small" @click="applyFilters" :loading="store.loading" block>刷新</NButton>
      </NSpace>
    </div>

    <div class="meta">共 {{ store.total }} 篇</div>

    <NScrollbar class="scroll">
      <NEmpty v-if="!store.loading && store.items.length === 0" description="暂无论文，先运行 ingest" />
      <NCard
        v-for="p in store.items"
        :key="p.paper_id"
        size="small"
        class="paper-card"
        :title="p.title"
        hoverable
        @click="openArxiv(p.paper_id)"
      >
        <template #header-extra>
          <NTag size="tiny" :bordered="false" type="info">{{ p.year }}</NTag>
        </template>
        <NSpace :size="4" :wrap-item="false" class="tags">
          <NTag size="tiny" :bordered="false">{{ p.primary_category }}</NTag>
          <NTag size="tiny" :bordered="false" type="success">{{ p.num_chunks }} chunks</NTag>
        </NSpace>
        <div class="authors" v-if="p.authors?.length">
          {{ p.authors.slice(0, 3).join(', ') }}<span v-if="p.authors.length > 3"> 等</span>
        </div>
      </NCard>
    </NScrollbar>
  </div>
</template>

<style scoped>
.paper-list {
  display: flex;
  flex-direction: column;
  height: 100%;
  padding: 12px;
  gap: 8px;
  overflow: hidden;
}
.filters { flex: 0 0 auto; }
.meta { font-size: 12px; opacity: 0.6; }
.scroll { flex: 1 1 auto; min-height: 0; }
.paper-card { margin-bottom: 8px; cursor: pointer; }
.paper-card :deep(.n-card-header__main) { font-size: 13px; line-height: 1.4; }
.tags { margin: 4px 0; }
.authors { font-size: 12px; opacity: 0.7; margin-top: 4px; }
</style>
