<script setup lang="ts">
import PaperList from '../components/PaperList.vue'
import ChatWindow from '../components/ChatWindow.vue'
import CitationCard from '../components/CitationCard.vue'
</script>

<template>
  <div class="layout">
    <aside class="left"><PaperList /></aside>
    <main class="center"><ChatWindow /></main>
    <aside class="right"><CitationCard /></aside>
  </div>
</template>

<style scoped>
.layout {
  display: grid;
  grid-template-columns: 300px 1fr 360px;
  height: 100vh;
  background: #0f1115;
}
.left { border-right: 1px solid #1f2937; overflow: hidden; }
.center { overflow: hidden; }
.right { border-left: 1px solid #1f2937; overflow: hidden; }

@media (max-width: 1200px) {
  .layout { grid-template-columns: 260px 1fr 320px; }
}
@media (max-width: 960px) {
  .layout { grid-template-columns: 1fr; grid-template-rows: auto 1fr auto; }
  .left, .right { border: none; max-height: 30vh; }
}
</style>
