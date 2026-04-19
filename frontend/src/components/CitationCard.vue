<script setup lang="ts">
import { NCard, NEmpty, NScrollbar, NTag, NSpace } from 'naive-ui'
import { useChatStore } from '../stores/chat'

const chatStore = useChatStore()

function openArxiv(id: string) {
  window.open(`https://arxiv.org/abs/${id}`, '_blank')
}
</script>

<template>
  <div class="cites">
    <div class="head">引用来源</div>
    <NScrollbar class="scroll">
      <NEmpty v-if="!chatStore.currentSources.length" description="发送问题后在此查看引用" />
      <NCard
        v-for="s in chatStore.currentSources"
        :key="s.paper_id + ':' + (s.chunk_index ?? 0)"
        size="small"
        class="ref-card"
        hoverable
        @click="openArxiv(s.paper_id)"
      >
        <template #header>
          <div class="ref-title">{{ s.title || s.paper_id }}</div>
        </template>
        <template #header-extra>
          <NTag size="tiny" type="info">{{ s.paper_id }}</NTag>
        </template>
        <NSpace :size="6" class="meta">
          <NTag size="tiny" :bordered="false">p.{{ s.page_num ?? '?' }}</NTag>
          <NTag size="tiny" :bordered="false" v-if="s.primary_category">{{ s.primary_category }}</NTag>
          <NTag size="tiny" :bordered="false" v-if="s.year">{{ s.year }}</NTag>
          <NTag size="tiny" :bordered="false" type="warning" v-if="s.score != null">score {{ s.score.toFixed(3) }}</NTag>
        </NSpace>
        <div class="authors" v-if="s.authors?.length">
          {{ s.authors.slice(0, 3).join(', ') }}<span v-if="s.authors.length > 3"> 等</span>
        </div>
        <div class="snippet" v-if="s.snippet">{{ s.snippet }}</div>
      </NCard>
    </NScrollbar>
  </div>
</template>

<style scoped>
.cites {
  display: flex;
  flex-direction: column;
  height: 100%;
  padding: 12px;
  gap: 8px;
  overflow: hidden;
}
.head { font-size: 13px; font-weight: 600; opacity: 0.85; }
.scroll { flex: 1 1 auto; min-height: 0; }
.ref-card { margin-bottom: 10px; cursor: pointer; }
.ref-title { font-size: 13px; line-height: 1.4; font-weight: 500; }
.meta { margin: 4px 0; }
.authors { font-size: 12px; opacity: 0.75; margin-top: 2px; }
.snippet {
  font-size: 12px;
  line-height: 1.5;
  opacity: 0.75;
  margin-top: 6px;
  border-left: 2px solid #374151;
  padding-left: 8px;
}
</style>
