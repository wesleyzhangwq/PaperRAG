<template>
  <div class="flex justify-end">
    <div class="max-w-[80%] px-4 py-3 rounded-card bg-accent-light text-text-primary text-sm whitespace-pre-wrap break-words">
      {{ content }}
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{ content: string }>()
</script>
