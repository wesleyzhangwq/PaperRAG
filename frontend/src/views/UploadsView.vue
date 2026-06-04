<template>
  <div class="flex-1 overflow-y-auto bg-bg-primary">
    <section class="max-w-3xl mx-auto px-6 py-6 space-y-5">
      <header>
        <h2 class="text-lg font-semibold text-text-primary">上传管理</h2>
        <p class="text-sm text-text-tertiary mt-1">上传 PDF 后会自动入库并生成检索 chunks。</p>
      </header>

      <form class="rounded-card border border-border bg-bg-card p-4 space-y-4" @submit.prevent="submit">
        <div>
          <label class="block text-xs font-medium text-text-tertiary mb-1">论文标题</label>
          <input
            v-model="title"
            class="w-full px-3 py-2 rounded-md border border-border bg-bg-primary text-sm outline-none focus:border-accent"
            placeholder="可选，默认使用文件名"
          />
        </div>
        <div>
          <label class="block text-xs font-medium text-text-tertiary mb-1">PDF 文件</label>
          <input
            type="file"
            accept="application/pdf,.pdf"
            class="block w-full text-sm text-text-secondary"
            @change="onFile"
          />
        </div>
        <div class="flex items-center justify-between gap-3">
          <p class="text-xs text-text-tertiary">{{ statusText }}</p>
          <button
            type="submit"
            class="px-3 py-2 rounded-md bg-accent text-white text-sm hover:opacity-90 disabled:opacity-50"
            :disabled="!file || uploading"
          >上传并入库</button>
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
                {{ item.paper_id }} · {{ item.num_chunks }} chunks · {{ item.message || item.status }}
              </p>
            </div>
            <span
              class="px-2 py-1 rounded-md text-xs"
              :class="statusClass(item.status)"
            >{{ item.status }}</span>
          </div>
        </article>
      </section>
    </section>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { getUploadJob, listUploadJobs, uploadPdf } from '../api/uploads'
import type { UploadJob } from '../types'

const title = ref('')
const file = ref<File | null>(null)
const uploading = ref(false)
const error = ref('')
const jobs = ref<UploadJob[]>([])

const statusText = computed(() => {
  if (uploading.value) return '正在上传和解析，较大的 PDF 可能需要几十秒。'
  if (file.value) return `${file.value.name} · ${Math.round(file.value.size / 1024 / 1024 * 10) / 10} MB`
  return '请选择一个 PDF 文件。'
})

onMounted(loadJobs)

async function loadJobs() {
  try {
    const data = await listUploadJobs({ limit: 20 })
    jobs.value = data.items
  } catch {
    // Upload history is useful but should not block the page.
  }
}

function onFile(event: Event) {
  const input = event.target as HTMLInputElement
  file.value = input.files?.[0] || null
  error.value = ''
}

async function submit() {
  if (!file.value || uploading.value) return
  uploading.value = true
  error.value = ''
  try {
    const result = await uploadPdf(file.value, title.value)
    if (result.job_id) {
      jobs.value = [{
        job_id: result.job_id,
        paper_id: result.paper_id,
        filename: file.value.name,
        title: title.value || file.value.name.replace(/\.pdf$/i, ''),
        status: result.status,
        num_chunks: result.num_chunks,
        message: result.message,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      }, ...jobs.value]
      pollJob(result.job_id)
    }
    title.value = ''
    file.value = null
  } catch {
    error.value = '上传失败。请确认文件是 PDF，大小不超过后端限制，并稍后重试。'
  } finally {
    uploading.value = false
  }
}

async function pollJob(jobId: string) {
  for (let i = 0; i < 60; i += 1) {
    await new Promise(resolve => setTimeout(resolve, 1500))
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
}

function statusClass(status: string) {
  if (status === 'failed') return 'bg-red-50 text-red-700'
  if (status === 'queued' || status === 'running') return 'bg-yellow-50 text-yellow-700'
  return 'bg-green-50 text-green-700'
}
</script>
