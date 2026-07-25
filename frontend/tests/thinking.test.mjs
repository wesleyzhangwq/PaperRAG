import assert from 'node:assert/strict'
import test from 'node:test'

import {
  applyPlanEvent,
  applyStageEvent,
  presentationToTimeline,
} from '../src/utils/timeline.ts'

test('presentation replay preserves stage and retrieval debug details', () => {
  const presentation = {
    answer: '',
    confidence: 'low',
    confidence_reason: '',
    sources: [],
    retrieval_summary: {
      total_chunks: 8,
      total_papers: 5,
      main_topics: [],
      is_fallback: true,
      narrative: '',
    },
    steps: [
      {
        index: 0,
        name: '意图分析',
        action: 'intent_analysis',
        status: 'completed',
        user_message: '已理解你问题的意图与重点。',
        duration_ms: 100,
        debug: {
          tool: 'intent_analysis',
          params: {},
          reason: '',
          raw_summary: 'type=simple',
          extra: { raw_summary: 'type=simple' },
        },
      },
      {
        index: 1,
        name: '制定检索计划',
        action: 'planning',
        status: 'completed',
        user_message: '已制定检索与生成计划。',
        duration_ms: 100,
        debug: {
          tool: 'planning',
          params: {},
          reason: '',
          raw_summary: 'generated 3 steps',
          extra: { raw_summary: 'generated 3 steps' },
        },
      },
      {
        index: 2,
        name: '检索相关论文',
        action: 'retrieve_local',
        status: 'warning',
        user_message: '找到 8 个文献片段。',
        duration_ms: 529,
        debug: {
          tool: 'retrieve_local',
          params: {
            query: 'Deep Learning Foundations 里哪些论文奠定了现代大模型基础？',
            top_k: 8,
          },
          reason: 'fallback',
          raw_summary: 'found 8 chunks (fallback)',
          extra: {
            raw_summary: 'found 8 chunks (fallback)',
            hits: [{ paper_id: '2402.05424', title: 'Neural Circuit Diagrams' }],
            total: 8,
            query_used: 'Deep Learning Foundations 里哪些论文奠定了现代大模型基础？',
            used_fallback: true,
          },
        },
      },
    ],
  }
  const timeline = presentationToTimeline(presentation)

  assert.equal(timeline[0].kind, 'stage')
  assert.deepEqual(timeline[0].detail, { raw_summary: 'type=simple' })
  assert.equal(timeline[2].kind, 'step')
  assert.equal(timeline[2].status, 'warning')
  assert.equal(timeline[2].durationMs, 529)
  assert.equal(timeline[2].detail?.query_used, 'Deep Learning Foundations 里哪些论文奠定了现代大模型基础？')
  assert.equal(timeline[2].detail?.total, 8)
  assert.deepEqual(timeline[2].detail?.hits, [
    { paper_id: '2402.05424', title: 'Neural Circuit Diagrams' },
  ])
})

test('stable stage ids update plan items without duplicating or downgrading them', () => {
  const planned = applyPlanEvent([], {
    revision: 1,
    steps: [
      {
        id: 'step:0',
        action: 'retrieve_local',
        title: '检索本地论文',
        reason: '优先使用本地证据',
      },
    ],
  })
  const running = applyStageEvent(planned, {
    id: 'step:0',
    stage: 'retrieve_step',
    status: 'start',
    title: '检索本地论文',
  })
  const republished = applyPlanEvent(running, {
    revision: 2,
    steps: [
      {
        id: 'step:0',
        action: 'retrieve_local',
        title: '检索本地论文（已路由）',
        reason: '保持本地优先',
      },
    ],
  })
  const done = applyStageEvent(republished, {
    id: 'step:0',
    stage: 'retrieve_step',
    status: 'done',
    title: '检索本地论文（已路由）',
    summary: '找到 8 个片段',
    duration_ms: 529,
  })

  assert.equal(done.length, 1)
  assert.equal(done[0].status, 'done')
  assert.equal(done[0].title, '检索本地论文（已路由）')
  assert.equal(done[0].summary, '找到 8 个片段')
  assert.equal(done[0].durationMs, 529)
})
