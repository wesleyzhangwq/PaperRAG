<template>
  <div class="border-t border-border bg-bg-card px-4 py-3">
    <div class="max-w-3xl mx-auto flex items-end gap-2">
      <textarea
        ref="inputRef"
        v-model="input"
        @keydown.enter.exact.prevent="send"
        :disabled="disabled"
        placeholder="输入你的问题..."
        rows="1"
        class="flex-1 resize-none rounded-sm border border-border bg-bg-primary px-3 py-2 text-sm text-text-primary placeholder:text-text-tertiary focus:outline-none focus:border-accent transition"
      />
      <button
        @click="send"
        :disabled="disabled || !input.trim()"
        class="px-4 py-2 rounded-sm bg-accent text-white text-sm font-medium disabled:opacity-40 hover:opacity-90 transition"
      >
        发送
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { nextTick, ref, watch } from 'vue'

const props = defineProps<{ disabled: boolean; draft?: string }>()
const emit = defineEmits<{ send: [query: string] }>()

const input = ref('')
const inputRef = ref<HTMLTextAreaElement>()

watch(
  () => props.draft,
  async (draft) => {
    if (!draft) return
    input.value = draft
    await nextTick()
    inputRef.value?.focus()
  },
)

function send() {
  if (!input.value.trim() || props.disabled) return
  emit('send', input.value.trim())
  input.value = ''
}
</script>
