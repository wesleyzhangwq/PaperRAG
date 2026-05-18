import { ref } from 'vue'
import type { ThinkingStep, SSEPlan, StepTrace } from '../types'

export function useThinking() {
  const steps = ref<ThinkingStep[]>([])
  const isThinking = ref(false)

  function startFromPlan(plan: SSEPlan) {
    isThinking.value = true
    steps.value = plan.steps.map((s, i) => ({
      index: i,
      action: s.action,
      reason: s.reason,
      status: 'pending' as const,
    }))
  }

  function markStepStart(index: number) {
    const step = steps.value.find(s => s.index === index)
    if (step) {
      step.status = 'running'
    }
  }

  function markStepDone(trace: StepTrace) {
    // Find the currently running step (or first pending as fallback)
    const step = steps.value.find(s => s.status === 'running')
      || steps.value.find(s => s.status === 'pending')
    if (step) {
      step.status = 'done'
      step.outputSummary = trace.output_summary
      step.durationMs = trace.duration_ms
    }
  }

  function addExtraSteps(newSteps: { action: string; reason: string }[]) {
    const maxIdx = steps.value.length
    const extra = newSteps.map((s, i) => ({
      index: maxIdx + i,
      action: s.action,
      reason: s.reason,
      status: 'pending' as const,
    }))
    steps.value.push(...extra)
  }

  function markFailed() {
    const step = steps.value.find(s => s.status === 'running')
    if (step) step.status = 'failed'
  }

  function finish() {
    isThinking.value = false
    // Mark any remaining pending steps as done (skipped)
    steps.value.forEach(s => {
      if (s.status === 'pending') s.status = 'done'
    })
  }

  function reset() {
    steps.value = []
    isThinking.value = false
  }

  return { steps, isThinking, startFromPlan, markStepStart, markStepDone, addExtraSteps, markFailed, finish, reset }
}
