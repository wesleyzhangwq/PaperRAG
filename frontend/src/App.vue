<script setup lang="ts">
import { NConfigProvider, NMessageProvider, darkTheme } from 'naive-ui'
import HomeView from './views/HomeView.vue'
</script>

<template>
  <NConfigProvider :theme="darkTheme">
    <NMessageProvider>
      <HomeView />
    </NMessageProvider>
  </NConfigProvider>
</template>
