<template>
  <span class="inline-flex gap-1">
    <span v-for="i in 3" :key="i" class="w-1.5 h-1.5 rounded-full bg-text-tertiary animate-bounce" :style="{ animationDelay: `${i * 150}ms` }" />
  </span>
</template>
