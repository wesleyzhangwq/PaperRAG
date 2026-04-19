<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue'
import { NButton, NInput, NScrollbar, NSpin, NUpload, NUploadTrigger, useMessage, type UploadCustomRequestOptions } from 'naive-ui'
import { marked } from 'marked'
import { usePapersStore } from '../stores/papers'
import { useChatStore } from '../stores/chat'
import { uploadPdf } from '../api/client'

const chatStore = useChatStore()
const papersStore = usePapersStore()
const message = useMessage()

const input = ref('')
const scrollRef = ref<any>(null)

const mdMessages = computed(() =>
  chatStore.messages.map((m) => ({ ...m, html: marked.parse(m.content) as string })),
)

watch(
  () => chatStore.messages.length,
  async () => {
    await nextTick()
    scrollRef.value?.scrollTo({ top: 99999, behavior: 'smooth' })
  },
)

async function send() {
  const q = input.value.trim()
  if (!q || chatStore.loading) return
  input.value = ''
  const filter = {
    category: papersStore.filters.category ?? undefined,
    year_min: papersStore.filters.year_min ?? undefined,
    year_max: papersStore.filters.year_max ?? undefined,
  }
  await chatStore.ask(q, filter)
}

async function handleUpload({ file, onFinish, onError }: UploadCustomRequestOptions) {
  try {
    const f = file.file as File
    const r = await uploadPdf(f)
    if (r.status === 'ok') {
      message.success(`已入库：${r.paper_id}（${r.num_chunks} chunks）`)
      papersStore.load()
      onFinish()
    } else {
      message.error(`失败：${r.message ?? r.status}`)
      onError()
    }
  } catch (e: any) {
    message.error(e?.message ?? 'upload failed')
    onError()
  }
}

const quickQueries = [
  'Transformer 的最新优化方向？',
  '检索增强生成（RAG）的最新进展？',
  'LLM 推理效率的典型优化方法？',
  'Agent 相关工作有哪些？',
  '大模型微调的常见方法？',
]
</script>

<template>
  <div class="chat-window">
    <div class="header">
      <div class="title">PaperRAG · 对话</div>
      <NSpin v-if="chatStore.loading" size="small" />
      <div class="actions">
        <NUpload :custom-request="handleUpload" accept=".pdf" :show-file-list="false">
          <NUploadTrigger abstract>
            <NButton size="small" secondary>上传 PDF</NButton>
          </NUploadTrigger>
        </NUpload>
        <NButton size="small" quaternary @click="chatStore.clear" :disabled="!chatStore.messages.length">清空</NButton>
      </div>
    </div>

    <NScrollbar ref="scrollRef" class="messages">
      <div v-if="chatStore.messages.length === 0" class="welcome">
        <h2>欢迎使用 PaperRAG</h2>
        <p>基于 arXiv 最新论文的本地 RAG 问答。试试以下问题：</p>
        <div class="quick">
          <NButton v-for="q in quickQueries" :key="q" size="small" tertiary @click="input = q">{{ q }}</NButton>
        </div>
      </div>
      <div
        v-for="m in mdMessages"
        :key="m.id"
        class="msg"
        :class="m.role"
      >
        <div class="role">{{ m.role === 'user' ? '你' : '助手' }}</div>
        <div class="bubble">
          <div class="md-body" v-html="m.html" />
          <div v-if="m.role === 'assistant' && m.sources?.length" class="refs">
            <span class="refs-title">参考 ({{ m.used_chunks ?? m.sources.length }} chunks)：</span>
            <a
              v-for="s in m.sources"
              :key="s.paper_id"
              :href="s.arxiv_url ?? `https://arxiv.org/abs/${s.paper_id}`"
              target="_blank"
              class="ref-chip"
              :title="s.title"
            >
              [{{ s.paper_id }}]
            </a>
          </div>
        </div>
      </div>
    </NScrollbar>

    <div class="composer">
      <NInput
        v-model:value="input"
        type="textarea"
        :autosize="{ minRows: 2, maxRows: 6 }"
        placeholder="询问关于论文的任何问题（Ctrl/Cmd + Enter 发送）"
        @keydown.enter.ctrl.prevent="send"
        @keydown.enter.meta.prevent="send"
      />
      <NButton type="primary" @click="send" :loading="chatStore.loading" :disabled="!input.trim()">
        发送
      </NButton>
    </div>
  </div>
</template>

<style scoped>
.chat-window {
  display: flex;
  flex-direction: column;
  height: 100%;
  padding: 12px 16px;
  gap: 10px;
  overflow: hidden;
}
.header {
  display: flex;
  align-items: center;
  gap: 12px;
  flex: 0 0 auto;
  padding-bottom: 8px;
  border-bottom: 1px solid #1f2937;
}
.title { font-weight: 600; font-size: 16px; }
.actions { margin-left: auto; display: flex; gap: 8px; }

.messages { flex: 1 1 auto; min-height: 0; }

.welcome {
  padding: 32px 16px;
  opacity: 0.85;
}
.welcome h2 { margin: 0 0 6px; }
.welcome .quick { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }

.msg { padding: 10px 4px; }
.msg .role {
  font-size: 12px;
  font-weight: 600;
  opacity: 0.7;
  margin-bottom: 4px;
}
.msg.user .role { color: #93c5fd; }
.msg.assistant .role { color: #6ee7b7; }
.msg .bubble {
  background: #1a1e27;
  padding: 10px 14px;
  border-radius: 8px;
  max-width: 100%;
}
.msg.user .bubble { background: #1e3a5f; }

.refs {
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px dashed #374151;
  font-size: 12px;
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  align-items: center;
}
.refs-title { opacity: 0.7; }
.ref-chip {
  color: #60a5fa;
  text-decoration: none;
  background: #1f2937;
  padding: 2px 6px;
  border-radius: 4px;
  font-family: monospace;
}
.ref-chip:hover { background: #2d3a4d; }

.composer {
  display: flex;
  gap: 8px;
  align-items: flex-end;
  flex: 0 0 auto;
}
.composer :deep(.n-input) { flex: 1 1 auto; }
</style>
