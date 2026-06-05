import type {
  Presentation,
  ThinkingStep,
  ToolCallEvent,
  ToolResultEvent,
} from '../types'

export interface ResolvedStepDetails {
  params?: Record<string, unknown>
  result?: Record<string, unknown>
}

export function presentationStepsToThinking(presentation: Presentation): ThinkingStep[] {
  return (presentation.steps || []).map((step, index) => ({
    index,
    action: step.action,
    reason: step.debug?.reason || '',
    status: step.status === 'error' ? 'failed' : 'done',
    outputSummary: step.debug?.raw_summary || step.user_message,
    durationMs: step.duration_ms,
    detailParams: objectWithEntries(step.debug?.params) ? step.debug.params : undefined,
    detailResult: objectWithEntries(step.debug?.extra) ? step.debug.extra : undefined,
    detailSource: 'presentation',
  }))
}

export function resolveStepDetails(
  step: ThinkingStep,
  call?: ToolCallEvent,
  result?: ToolResultEvent,
): ResolvedStepDetails {
  const stepParams = objectWithEntries(step.detailParams) ? step.detailParams : undefined
  const stepResult = objectWithEntries(step.detailResult) ? step.detailResult : undefined
  if (step.detailSource === 'presentation') {
    return {
      params: stepParams,
      result: stepResult,
    }
  }
  return {
    params: stepParams || (objectWithEntries(call?.params) ? call?.params : undefined),
    result: stepResult || (objectWithEntries(result?.detail) ? result?.detail : undefined),
  }
}

export function hasResolvedStepDetails(details: ResolvedStepDetails): boolean {
  return objectWithEntries(details.params) || objectWithEntries(details.result)
}

function objectWithEntries(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === 'object' && !Array.isArray(value)
    && Object.keys(value as Record<string, unknown>).length > 0
}
