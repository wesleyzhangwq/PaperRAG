import { defineStore } from 'pinia'
import { ref } from 'vue'
import { listPapers, type PaperSummary } from '../api/client'

export const usePapersStore = defineStore('papers', () => {
  const items = ref<PaperSummary[]>([])
  const total = ref(0)
  const loading = ref(false)

  const filters = ref<{
    category: string | null
    year_min: number | null
    year_max: number | null
    q: string
  }>({
    category: null,
    year_min: null,
    year_max: null,
    q: '',
  })

  async function load() {
    loading.value = true
    try {
      const r = await listPapers({
        category: filters.value.category ?? undefined,
        year_min: filters.value.year_min ?? undefined,
        year_max: filters.value.year_max ?? undefined,
        q: filters.value.q || undefined,
        limit: 200,
      })
      items.value = r.items
      total.value = r.total
    } finally {
      loading.value = false
    }
  }

  return { items, total, loading, filters, load }
})
