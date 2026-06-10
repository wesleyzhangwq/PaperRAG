import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import * as convApi from '../api/conversations'
import { presentationToTimeline, tracesToTimeline } from '../utils/timeline'
import type { Conversation, Message, ServerMessage } from '../types'

const ACTIVE_KEY = 'paperrag.activeConversation'

function serverMessagesToLocal(rows: ServerMessage[]): Message[] {
  return rows.map(r => {
    const timeline = r.presentation?.steps?.length
      ? presentationToTimeline(r.presentation)
      : tracesToTimeline(r.thinking || [])
    return {
      id: `srv-${r.id}`,
      role: r.role,
      content: r.content,
      sources: r.sources || [],
      timeline,
      presentation: r.presentation || null,
      elapsedMs: r.elapsed_ms ?? deriveElapsedMs(timeline),
      timestamp: r.created_at ? new Date(r.created_at).getTime() : Date.now(),
    }
  })
}

function deriveElapsedMs(timeline: Message['timeline']): number {
  return Math.round((timeline || []).reduce((sum, item) => sum + (item.durationMs || 0), 0))
}

export const useConversationsStore = defineStore('conversations', () => {
  // metadata for all conversations (sorted: pinned first, then updated_at desc)
  const conversations = ref<Conversation[]>([])
  // per-conversation message arrays, in-memory cache keyed by conversation id
  const messagesByConv = ref<Record<string, Message[]>>({})
  // currently selected conversation id
  const activeId = ref<string>('')
  const loading = ref(false)

  const pinned = computed(() => conversations.value.filter(c => c.pinned))
  const recent = computed(() => conversations.value.filter(c => !c.pinned))
  const active = computed(() => conversations.value.find(c => c.id === activeId.value) || null)
  const activeMessages = computed(() => messagesByConv.value[activeId.value] || [])

  async function loadAll() {
    loading.value = true
    try {
      const list = await convApi.listConversations()
      conversations.value = list
      // restore last active from localStorage if still present
      const stored = window.localStorage.getItem(ACTIVE_KEY) || ''
      if (stored && list.some(c => c.id === stored)) {
        await selectConversation(stored)
      } else if (list.length > 0) {
        await selectConversation(list[0].id)
      } else {
        await createNew()
      }
    } finally {
      loading.value = false
    }
  }

  async function selectConversation(id: string) {
    activeId.value = id
    try { window.localStorage.setItem(ACTIVE_KEY, id) } catch { /* ignore */ }
    if (!messagesByConv.value[id]) {
      try {
        const rows = await convApi.getMessages(id)
        messagesByConv.value[id] = serverMessagesToLocal(rows)
      } catch {
        messagesByConv.value[id] = []
      }
    }
  }

  async function createNew(): Promise<string> {
    const conv = await convApi.createConversation()
    // prepend so it appears at top of Recent
    conversations.value = [conv, ...conversations.value.filter(c => c.id !== conv.id)]
    messagesByConv.value[conv.id] = []
    await selectConversation(conv.id)
    return conv.id
  }

  async function togglePin(id: string) {
    const conv = conversations.value.find(c => c.id === id)
    if (!conv) return
    const updated = await convApi.updateConversation(id, { pinned: !conv.pinned })
    Object.assign(conv, updated)
    // resort: pinned first, then by updated_at desc within group
    sortLocal()
  }

  async function rename(id: string, title: string) {
    const updated = await convApi.updateConversation(id, { title })
    const idx = conversations.value.findIndex(c => c.id === id)
    if (idx >= 0) conversations.value[idx] = updated
  }

  async function remove(id: string) {
    await convApi.deleteConversation(id)
    conversations.value = conversations.value.filter(c => c.id !== id)
    delete messagesByConv.value[id]
    if (activeId.value === id) {
      if (conversations.value.length > 0) {
        await selectConversation(conversations.value[0].id)
      } else {
        await createNew()
      }
    }
  }

  // --- local message manipulation (called by useChat during streaming) ---
  function appendMessage(msg: Message) {
    const id = activeId.value
    if (!id) return
    const arr = messagesByConv.value[id] || []
    messagesByConv.value[id] = [...arr, msg]
  }

  function updateLastAssistant(patch: Partial<Message>) {
    const id = activeId.value
    if (!id) return
    const arr = messagesByConv.value[id] || []
    if (arr.length === 0) return
    const last = arr[arr.length - 1]
    if (last.role !== 'assistant') return
    Object.assign(last, patch)
    messagesByConv.value[id] = [...arr]
  }

  function bumpActive(title?: string) {
    const conv = conversations.value.find(c => c.id === activeId.value)
    if (!conv) return
    if (title && (conv.title === '新对话' || !conv.title)) {
      conv.title = title.slice(0, 60)
      convApi.updateConversation(conv.id, { title: conv.title }).catch(() => { /* ignore */ })
    }
    conv.updated_at = new Date().toISOString()
    sortLocal()
  }

  function sortLocal() {
    conversations.value = [
      ...conversations.value.filter(c => c.pinned).sort(byUpdated),
      ...conversations.value.filter(c => !c.pinned).sort(byUpdated),
    ]
  }

  function byUpdated(a: Conversation, b: Conversation) {
    return (b.updated_at || '').localeCompare(a.updated_at || '')
  }

  return {
    conversations, pinned, recent, active, activeId, activeMessages,
    messagesByConv, loading,
    loadAll, selectConversation, createNew, togglePin, rename, remove,
    appendMessage, updateLastAssistant, bumpActive,
  }
})
