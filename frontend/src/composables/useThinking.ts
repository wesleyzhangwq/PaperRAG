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
      status: i === 0 ? 'running' : 'pending',
    }))
  }

  function markStepDone(trace: StepTrace) {
    const step = steps.value.find(s => s.status === 'running')
    if (step) {
      step.status = 'done'
      step.outputSummary = trace.output_summary
      step.durationMs = trace.duration_ms
    }
    const next = steps.value.find(s => s.status === 'pending')
    if (next) next.status = 'running'
  }

  function markFailed() {
    const step = steps.value.find(s => s.status === 'running')
    if (step) step.status = 'failed'
  }

  function reset() {
    steps.value = []
    isThinking.value = false
  }

  return { steps, isThinking, startFromPlan, markStepDone, markFailed, reset }
}
