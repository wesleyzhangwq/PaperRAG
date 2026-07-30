import assert from 'node:assert/strict'
import test from 'node:test'

import { extractCitedIds, renderMarkdown } from '../src/utils/markdown.ts'

test('uploaded and arxiv citations render through the same pill protocol', () => {
  const text = (
    'Published [arxiv:1706.03762] and uploaded '
    + '[source:local-a1b2c3] evidence.'
  )
  const sources = [
    {
      paper_id: '1706.03762',
      title: 'Attention',
      authors: [],
      year: 2017,
      arxiv_url: 'https://arxiv.org/abs/1706.03762',
    },
    {
      paper_id: 'local-a1b2c3',
      title: 'Uploaded Notes',
      authors: [],
      year: 2026,
      source_kind: 'upload',
      arxiv_url: null,
    },
  ]

  const html = renderMarkdown(text, sources)

  assert.deepEqual(extractCitedIds(text), ['1706.03762', 'local-a1b2c3'])
  assert.match(html, /data-paper-id="1706\.03762"/)
  assert.match(html, /data-paper-id="local-a1b2c3"/)
})
