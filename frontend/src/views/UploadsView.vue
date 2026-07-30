<template>
  <div class="flex-1 overflow-y-auto bg-bg-primary">
    <section class="max-w-3xl mx-auto px-6 py-6 space-y-5">
      <header>
        <h2 class="text-lg font-semibold text-text-primary">上传管理</h2>
        <p class="text-sm text-text-tertiary mt-1">arXiv 与本地异构文件共用解析、分块、向量索引和可追溯入库链路。</p>
      </header>

      <form class="rounded-card border border-border bg-bg-card p-4 space-y-4" @submit.prevent="submitFiles">
        <div>
          <label class="block text-xs font-medium text-text-tertiary mb-1">本地文件</label>
          <label
            class="flex min-h-28 cursor-pointer flex-col items-center justify-center rounded-md border border-dashed border-border bg-bg-primary px-4 py-5 text-center hover:border-accent"
            @dragover.prevent
            @drop.prevent="onDrop"
          >
            <span class="text-sm font-medium text-text-primary">选择文件或拖放到这里</span>
            <span class="mt-1 text-xs text-text-tertiary">PDF、DOCX、PPTX、HTML、Markdown、TXT、CSV、XLSX、PNG/JPEG/WebP/TIFF</span>
            <input
              class="sr-only"
              type="file"
              multiple
              accept=".pdf,.docx,.pptx,.html,.htm,.md,.markdown,.txt,.csv,.xlsx,.png,.jpg,.jpeg,.webp,.tif,.tiff"
              @change="onFileChange"
            />
          </label>
        </div>
        <div v-if="selectedFiles.length" class="space-y-1">
          <p
            v-for="file in selectedFiles"
            :key="`${file.name}:${file.size}`"
            class="text-xs text-text-tertiary"
          >{{ file.name }} · {{ formatBytes(file.size) }}</p>
        </div>
        <div class="flex items-center justify-between gap-3">
          <p class="text-xs text-text-tertiary">图片和扫描 PDF 使用本地 OCR；表格、页码、幻灯片和工作表 locator 会随 chunk 入库。</p>
          <button
            type="submit"
            class="shrink-0 px-3 py-2 rounded-md bg-accent text-white text-sm hover:opacity-90 disabled:opacity-50"
            :disabled="selectedFiles.length === 0 || uploadingFiles"
          >{{ uploadingFiles ? '正在上传…' : `上传 ${selectedFiles.length || ''} 个文件` }}</button>
        </div>
      </form>

      <form class="rounded-card border border-border bg-bg-card p-4 space-y-4" @submit.prevent="submit">
        <div>
          <label class="block text-xs font-medium text-text-tertiary mb-1">arXiv ID 或 URL</label>
          <textarea
            v-model="arxivInput"
            rows="5"
            class="w-full px-3 py-2 rounded-md border border-border bg-bg-primary text-sm outline-none focus:border-accent"
            placeholder="2511.16043&#10;https://arxiv.org/abs/2508.07407&#10;arXiv:2512.04123v1"
            @input="error = ''"
          ></textarea>
        </div>
        <div class="flex items-center justify-between gap-3">
          <p class="text-xs text-text-tertiary">{{ statusText }}</p>
          <button
            type="submit"
            class="px-3 py-2 rounded-md bg-accent text-white text-sm hover:opacity-90 disabled:opacity-50"
            :disabled="!canSubmit || importing"
          >导入 arXiv 论文</button>
        </div>
      </form>

      <div v-if="error" class="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
        {{ error }}
      </div>

      <section v-if="jobs.length > 0" class="space-y-2">
        <h3 class="text-xs font-medium text-text-tertiary uppercase tracking-wide">最近任务</h3>
        <article
          v-for="item in jobs"
          :key="item.job_id"
          class="rounded-card border border-border bg-bg-card p-4"
        >
          <div class="flex items-center justify-between gap-3">
            <div class="min-w-0">
              <p class="text-sm font-medium text-text-primary truncate">{{ item.title || item.filename }}</p>
              <p class="text-xs text-text-tertiary">
                {{ item.source_kind }} · {{ item.paper_id }} · {{ item.num_chunks }} chunks
              </p>
              <p class="text-xs text-text-tertiary mt-0.5">
                {{ item.stage }} · {{ item.progress }}% · {{ item.message || item.status }}
              </p>
            </div>
            <span
              class="px-2 py-1 rounded-md text-xs"
              :class="statusClass(item.status)"
            >{{ item.status }}</span>
          </div>
          <div class="mt-3 h-1.5 overflow-hidden rounded-full bg-bg-primary">
            <div
              class="h-full rounded-full bg-accent transition-all duration-300"
              :style="{ width: `${Math.max(0, Math.min(100, item.progress || 0))}%` }"
            ></div>
          </div>
          <p v-if="item.warnings?.length" class="mt-2 text-xs text-yellow-700">
            {{ item.warnings.join('；') }}
          </p>
        </article>
      </section>
    </section>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { getUploadJob, importArxivPapers, listUploadJobs, uploadDocumentFiles } from '../api/uploads'
import type { UploadJob } from '../types'

const arxivInput = ref('')
const importing = ref(false)
const uploadingFiles = ref(false)
const selectedFiles = ref<File[]>([])
const error = ref('')
const jobs = ref<UploadJob[]>([])
const pollIntervalMs = 1500
const pollMaxAttempts = 800
const activePolls = new Set<string>()
let refreshTimer: number | undefined

const parsedArxivIds = computed(() => parseArxivInputs(arxivInput.value))
const canSubmit = computed(() => parsedArxivIds.value.length > 0)
const statusText = computed(() => {
  if (importing.value) return '正在创建导入任务，后台会继续下载和解析 PDF。'
  const count = parsedArxivIds.value.length
  if (count > 0) return `将导入 ${count} 篇 arXiv 论文；已存在的论文会自动跳过。`
  return '支持一次粘贴多个 ID、arXiv abs URL 或 PDF URL。'
})

onMounted(() => {
  loadJobs()
  refreshTimer = window.setInterval(loadJobs, 10000)
})

onBeforeUnmount(() => {
  if (refreshTimer) window.clearInterval(refreshTimer)
})

async function loadJobs() {
  try {
    const data = await listUploadJobs({ limit: 20 })
    jobs.value = data.items
    for (const job of data.items) {
      if (['queued', 'running'].includes(job.status)) {
        pollJob(job.job_id)
      }
    }
  } catch {
    // Upload history is useful but should not block the page.
  }
}

function parseArxivInputs(value: string): string[] {
  const seen = new Set<string>()
  const items: string[] = []
  for (const raw of value.split(/[\s,;，；]+/)) {
    const item = raw.trim()
    if (!item || seen.has(item)) continue
    seen.add(item)
    items.push(item)
  }
  return items
}

function onFileChange(event: Event) {
  selectedFiles.value = Array.from((event.target as HTMLInputElement).files || [])
  error.value = ''
}

function onDrop(event: DragEvent) {
  selectedFiles.value = Array.from(event.dataTransfer?.files || []).slice(0, 20)
  error.value = ''
}

function formatBytes(size: number) {
  if (size < 1024) return `${size} B`
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`
  return `${(size / 1024 / 1024).toFixed(1)} MB`
}

async function submitFiles() {
  if (!selectedFiles.value.length || uploadingFiles.value) return
  uploadingFiles.value = true
  error.value = ''
  try {
    const result = await uploadDocumentFiles(selectedFiles.value)
    for (const item of result.items) {
      if (item.job_id) pollJob(item.job_id)
    }
    selectedFiles.value = []
    await loadJobs()
  } catch (err) {
    const detail = (err as { response?: { data?: { detail?: { user_message?: string } } } })?.response?.data?.detail
    error.value = detail?.user_message || '文件上传失败，请检查格式、大小或服务状态。'
  } finally {
    uploadingFiles.value = false
  }
}

async function submit() {
  if (!canSubmit.value || importing.value) return
  importing.value = true
  error.value = ''
  try {
    const result = await importArxivPapers(parsedArxivIds.value)
    const now = new Date().toISOString()
    const newJobs = result.items
      .filter(item => item.job_id)
      .map(item => ({
        job_id: item.job_id as string,
        paper_id: item.paper_id,
        filename: `${item.paper_id}.pdf`,
        title: item.paper_id,
        status: item.status,
        stage: item.status === 'queued' ? 'queued' : 'completed',
        progress: item.status === 'queued' ? 0 : 100,
        num_chunks: item.num_chunks,
        message: item.message,
        source_kind: 'arxiv',
        media_type: 'application/pdf',
        warnings: [],
        created_at: now,
        updated_at: now,
      }))
    jobs.value = [...newJobs, ...jobs.value]
    for (const item of result.items) {
      if (item.job_id) pollJob(item.job_id)
    }
    arxivInput.value = ''
    await loadJobs()
  } catch (err) {
    const detail = (err as { response?: { data?: { detail?: { user_message?: string } } } })?.response?.data?.detail
    error.value = detail?.user_message || '导入失败。请确认 arXiv ID 或 URL 有效，并稍后重试。'
  } finally {
    importing.value = false
  }
}

async function pollJob(jobId: string) {
  if (activePolls.has(jobId)) return
  activePolls.add(jobId)
  try {
    for (let i = 0; i < pollMaxAttempts; i += 1) {
      await new Promise(resolve => setTimeout(resolve, pollIntervalMs))
      try {
        const job = await getUploadJob(jobId)
        const idx = jobs.value.findIndex(item => item.job_id === jobId)
        if (idx >= 0) jobs.value[idx] = job
        else jobs.value = [job, ...jobs.value]
        if (!['queued', 'running'].includes(job.status)) return
      } catch {
        return
      }
    }
    await loadJobs()
  } finally {
    activePolls.delete(jobId)
  }
}

function statusClass(status: string) {
  if (status === 'failed') return 'bg-red-50 text-red-700'
  if (status === 'queued' || status === 'running') return 'bg-yellow-50 text-yellow-700'
  return 'bg-green-50 text-green-700'
}
</script>
