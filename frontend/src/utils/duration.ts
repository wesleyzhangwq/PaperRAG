export function formatStepDuration(ms?: number | null): string {
  if (!ms) return ''
  if (ms < 1000) return `${Math.round(ms)}ms`
  return `${(ms / 1000).toFixed(2)}s`
}
