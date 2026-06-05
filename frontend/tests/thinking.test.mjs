import assert from 'node:assert/strict'
import test from 'node:test'

import {
  presentationStepsToThinking,
  resolveStepDetails,
} from '../src/utils/thinking.ts'

test('presentation step debug details take precedence over stale live tool indexes', () => {
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
  const staleCall = {
    index: 0,
    action: 'retrieve_local',
    params: { query: '', top_k: 8 },
    reason: 'fallback',
  }
  const staleResult = {
    index: 0,
    action: 'retrieve_local',
    duration_ms: 529,
    summary: 'found 8 chunks (fallback)',
    detail: {
      hits: [{ paper_id: 'wrong-step' }],
      total: 8,
    },
  }

  const steps = presentationStepsToThinking(presentation)
  const intentDetails = resolveStepDetails(steps[0], staleCall, staleResult)
  const retrieveDetails = resolveStepDetails(steps[2], undefined, undefined)

  assert.deepEqual(intentDetails.params, undefined)
  assert.deepEqual(intentDetails.result, { raw_summary: 'type=simple' })
  assert.equal(retrieveDetails.params?.query, 'Deep Learning Foundations 里哪些论文奠定了现代大模型基础？')
  assert.equal(retrieveDetails.result?.total, 8)
  assert.deepEqual(retrieveDetails.result?.hits, [
    { paper_id: '2402.05424', title: 'Neural Circuit Diagrams' },
  ])
})
