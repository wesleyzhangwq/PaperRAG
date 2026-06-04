<template>
  <div class="flex-1 overflow-y-auto bg-bg-primary">
    <section class="max-w-3xl mx-auto px-6 py-6 space-y-5">
      <header>
        <h2 class="text-lg font-semibold text-text-primary">设置</h2>
        <p class="text-sm text-text-tertiary mt-1">集中查看本地客户端和后端保护配置。</p>
      </header>

      <section class="rounded-card border border-border bg-bg-card p-4 space-y-3">
        <h3 class="text-sm font-medium text-text-primary">连接</h3>
        <div class="text-sm text-text-secondary space-y-1">
          <p>API 地址：<span class="font-mono text-xs">{{ apiBase }}</span></p>
          <p>流式协议：Server-Sent Events</p>
        </div>
      </section>

      <section class="rounded-card border border-border bg-bg-card p-4 space-y-3">
        <h3 class="text-sm font-medium text-text-primary">后端保护</h3>
        <ul class="space-y-2 text-sm text-text-secondary">
          <li>限流：后端通过 `RATE_LIMIT_*` 环境变量控制 chat/upload 请求频率。</li>
          <li>API Key：上线前设置 `API_AUTH_ENABLED=true` 和 `API_KEYS`。</li>
          <li>CORS：通过 `CORS_ORIGINS` 限制允许访问的前端域名。</li>
        </ul>
      </section>

      <section class="rounded-card border border-border bg-bg-card p-4 space-y-3">
        <h3 class="text-sm font-medium text-text-primary">界面偏好</h3>
        <p class="text-sm text-text-secondary">主题切换已在侧栏底部提供。多语言与完整可访问性仍在后续计划中。</p>
      </section>
    </section>
  </div>
</template>

<script setup lang="ts">
const apiBase = import.meta.env.VITE_API_BASE || 'http://localhost:8000'
</script>
