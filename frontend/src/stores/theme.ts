import { defineStore } from 'pinia'
import { ref, watch } from 'vue'
import type { Theme } from '../types'

const STORAGE_KEY = 'paperrag.theme'

function loadInitial(): Theme {
  if (typeof window === 'undefined') return 'light'
  const stored = window.localStorage.getItem(STORAGE_KEY) as Theme | null
  if (stored === 'light' || stored === 'dark') return stored
  // fall back to system preference once
  if (window.matchMedia?.('(prefers-color-scheme: dark)').matches) return 'dark'
  return 'light'
}

function applyTheme(theme: Theme) {
  if (typeof document === 'undefined') return
  const root = document.documentElement
  if (theme === 'dark') {
    root.classList.add('dark')
    root.setAttribute('data-theme', 'dark')
  } else {
    root.classList.remove('dark')
    root.setAttribute('data-theme', 'light')
  }
}

export const useThemeStore = defineStore('theme', () => {
  const theme = ref<Theme>(loadInitial())
  applyTheme(theme.value)

  function setTheme(t: Theme) {
    theme.value = t
  }

  function toggle() {
    theme.value = theme.value === 'light' ? 'dark' : 'light'
  }

  watch(theme, t => {
    applyTheme(t)
    try { window.localStorage.setItem(STORAGE_KEY, t) } catch { /* ignore */ }
  })

  return { theme, setTheme, toggle }
})
