import { defineStore } from 'pinia'
import { ref } from 'vue'

/** Ephemeral UI state for the in-flight chat turn. Actual messages live in
 *  the conversations store. */
export const useChatStore = defineStore('chat', () => {
  const isLoading = ref(false)
  const currentElapsedMs = ref(0)

  function reset() {
    isLoading.value = false
    currentElapsedMs.value = 0
  }

  return { isLoading, currentElapsedMs, reset }
})
