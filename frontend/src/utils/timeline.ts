import type {
  Presentation,
  SSEPlan,
  SSEStage,
  StepTrace,
  TimelineItem,
  TimelineStatus,
} from '../types'

/** Map a backend stage status onto the timeline vocabulary. */
function mapStatus(status: SSEStage['status']): TimelineStatus {
  switch (status) {
    case 'start': return 'running'
    case 'done': return 'done'
    case 'warning': return 'warning'
    case 'failed': return 'failed'
    case 'skipped': return 'skipped'
    default: return 'pending'
  }
}

/**
 * Upsert a stage event into the timeline by stable id.
 * Pure function: returns a new array (Vue reactivity-friendly).
 */
export function applyStageEvent(timeline: TimelineItem[], ev: SSEStage): TimelineItem[] {
  const next = [...timeline]
  const idx = next.findIndex(item => item.id === ev.id)
  const patch: Partial<TimelineItem> = {
    title: ev.title || next[idx]?.title || ev.stage,
    status: mapStatus(ev.status),
  }
  if (ev.summary) patch.summary = ev.summary
  if (ev.detail && Object.keys(ev.detail).length > 0) patch.detail = ev.detail
  if (typeof ev.duration_ms === 'number' && ev.status !== 'start') patch.durationMs = ev.duration_ms

  if (idx >= 0) {
    next[idx] = { ...next[idx], ...patch }
  } else {
    next.push({
      id: ev.id,
      kind: ev.stage === 'retrieve_step' ? 'step' : 'stage',
      title: ev.title || ev.stage,
      status: mapStatus(ev.status),
      summary: ev.summary,
      detail: ev.detail && Object.keys(ev.detail).length > 0 ? ev.detail : undefined,
      durationMs: ev.status !== 'start' ? ev.duration_ms : undefined,
    })
  }
  return next
}

/**
 * Merge a plan publication: ensure every step exists (pending), refresh
 * titles/reasons, never downgrade an already-running/done step.
 */
export function applyPlanEvent(timeline: TimelineItem[], plan: SSEPlan): TimelineItem[] {
  const next = [...timeline]
  for (const step of plan.steps || []) {
    const idx = next.findIndex(item => item.id === step.id)
    if (idx >= 0) {
      next[idx] = { ...next[idx], title: step.title || next[idx].title, reason: step.reason || next[idx].reason }
    } else {
      next.push({
        id: step.id,
        kind: 'step',
        title: step.title || step.action,
        status: 'pending',
        reason: step.reason,
      })
    }
  }
  return next
}

/** The item to surface in the collapsed header while running. */
export function currentActivity(timeline: TimelineItem[]): TimelineItem | null {
  for (let i = timeline.length - 1; i >= 0; i--) {
    if (timeline[i].status === 'running') return timeline[i]
  }
  return null
}

export function doneCount(timeline: TimelineItem[]): number {
  return timeline.filter(t => ['done', 'warning', 'failed', 'skipped'].includes(t.status)).length
}

// ---------------------------------------------------------------------------
// History replay converters
// ---------------------------------------------------------------------------

/** Rebuild a timeline from a persisted presentation payload (preferred). */
export function presentationToTimeline(presentation: Presentation): TimelineItem[] {
  return (presentation.steps || []).map((step, i) => ({
    id: `replay:${i}`,
    kind: isToolAction(step.action) ? 'step' as const : 'stage' as const,
    title: step.name || step.action,
    status: step.status === 'error' ? 'failed' as const
      : step.status === 'warning' ? 'warning' as const
      : 'done' as const,
    summary: step.user_message || step.debug?.raw_summary || '',
    reason: step.debug?.reason || '',
    detail: hasEntries(step.debug?.extra) ? step.debug.extra : undefined,
    durationMs: step.duration_ms,
  }))
}

/** Rebuild a timeline from raw step traces (legacy messages without presentation). */
export function tracesToTimeline(traces: StepTrace[]): TimelineItem[] {
  return (traces || []).map((t, i) => ({
    id: `replay:${i}`,
    kind: isToolAction(t.action) ? 'step' as const : 'stage' as const,
    title: t.action,
    status: 'done' as const,
    summary: t.output_summary,
    durationMs: t.duration_ms,
  }))
}

const TOOL_ACTIONS = new Set([
  'query_rewrite', 'retrieve_local', 'retrieve_arxiv', 'search_web',
  'get_paper_detail', 'get_paper_chunks', 'evaluate_docs',
])

function isToolAction(action: string): boolean {
  return TOOL_ACTIONS.has(action)
}

function hasEntries(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === 'object' && !Array.isArray(value)
    && Object.keys(value as Record<string, unknown>).length > 0
}
