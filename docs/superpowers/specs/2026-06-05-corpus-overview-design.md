# Corpus Overview Empty-State Design

## Goal

Show users what the current Cite Scope corpus roughly contains before they ask a question.

The overview appears in the new-conversation chat window, replacing the current empty state text. It should make the corpus feel inspectable and trustworthy without forcing users to open the paper library page.

## Non-Goals

- Do not add a new ingestion pipeline.
- Do not generate the overview with an LLM in the first version.
- Do not redesign the paper library page.
- Do not add background automations or scheduled jobs.
- Do not change retrieval, rerank, synthesis, or citation behavior.

## Product Behavior

When the active conversation has no messages, the chat window shows a `CorpusOverviewCard`.

The card contains:

1. **Corpus scale**
   - total ingested papers
   - total chunks
   - year range
   - top-level corpus description

2. **Topic map**
   - 5 to 7 topic groups derived from stored paper categories.
   - Each group shows a label, paper count, short description, and compact visual weight.

3. **Representative papers**
   - 2 to 3 paper titles per topic group.
   - Titles are selected deterministically from existing metadata, not generated.

4. **Suggested questions**
   - 4 to 6 clickable question chips.
   - Clicking a chip fills the input box. A later iteration may add direct-send, but fill-only is safer for the first version.

The overview disappears as soon as the conversation has messages.

## Data Model

Add a backend response model for `GET /papers/overview`:

```ts
interface CorpusOverviewResponse {
  total_papers: number
  total_chunks: number
  year_min: number | null
  year_max: number | null
  topic_buckets: CorpusTopicBucket[]
  suggested_questions: string[]
  generated_at: string
}

interface CorpusTopicBucket {
  key: string
  label: string
  description: string
  paper_count: number
  chunk_count: number
  representative_papers: CorpusRepresentativePaper[]
}

interface CorpusRepresentativePaper {
  paper_id: string
  title: string
  year: number | null
  primary_category: string
  arxiv_url: string
}
```

## Topic Bucketing

Use existing `Paper.categories` values as the primary signal. The current 500-paper corpus includes curated bucket labels such as:

- `llm_transformer`
- `rag_ir_memory`
- `agents_reasoning`
- `evaluation_factuality`
- `alignment_safety_eval`
- `multimodal_generative`
- `deep_learning`

Map known bucket keys to user-facing labels:

| Key | Label | Description |
|---|---|---|
| `rag_ir_memory` | RAG / Retrieval | Retrieval-augmented generation, dense retrieval, reranking, memory, evidence search |
| `agents_reasoning` | Agents / Reasoning | Tool use, planning, self-reflection, multi-step reasoning, agent workflows |
| `llm_transformer` | LLM / Transformer | Transformer models, pretraining, scaling, instruction tuning, model families |
| `evaluation_factuality` | Evaluation / Factuality | Benchmarks, hallucination, truthfulness, attribution, answer faithfulness |
| `alignment_safety_eval` | Alignment / Safety | RLHF, preference learning, safety evaluation, harmlessness, red teaming |
| `multimodal_generative` | Multimodal / Generative | Vision-language models, diffusion, image generation, multimodal instruction tuning |
| `deep_learning` | Deep Learning Foundations | Core optimization, representation learning, sequence models, neural architecture foundations |

If papers include only arXiv categories like `cs.CL` or `cs.AI`, group them under an `other` bucket labeled `Other AI Papers`.

Sort topic buckets by `paper_count` descending, then by the fixed priority above. Limit the visible list to 7 buckets.

## Representative Paper Selection

For each bucket:

1. Prefer papers with `ingest_status = "ok"` and `num_chunks > 0`.
2. Prefer papers whose `categories` contain the bucket key.
3. Sort by a deterministic score:
   - curated bucket match first
   - lower `year` gets a small boost for foundational coverage
   - higher `num_chunks` gets a small boost as a proxy for richer local context
4. Return up to 3 representative papers.

The first version does not need a manual “landmark score” column.

## Suggested Questions

Generate fixed questions from visible buckets using templates. Examples:

- `RAG / Retrieval 的技术路线是如何演进的？`
- `对比这批论文里的 dense retrieval、rerank 和 RAG 方法。`
- `Agents / Reasoning 相关论文主要解决了哪些问题？`
- `这批论文里关于 hallucination 和 factuality 的主要评估方法有哪些？`
- `Transformer 到现代 LLM 的关键转折点是什么？`

Return no more than 6 questions. Prefer questions tied to visible buckets. If the corpus is empty, return a small set of setup-oriented questions such as `如何上传第一篇论文？`.

## Backend Design

Add to `backend/app/routers/papers.py`:

```text
GET /papers/overview
```

The endpoint uses SQLAlchemy only:

- aggregate total paper count
- aggregate total chunk count
- aggregate year min/max
- load ingested papers with `paper_id`, `title`, `year`, `primary_category`, `categories`, `num_chunks`
- compute bucket counts and representatives in Python

The endpoint should not call Qdrant or the LLM. This keeps the overview cheap and stable.

## Frontend Design

Add:

```text
frontend/src/components/chat/CorpusOverviewCard.vue
frontend/src/api/papers.ts -> getCorpusOverview()
frontend/src/types/index.ts -> CorpusOverviewResponse types
```

Update:

```text
frontend/src/components/chat/MessageList.vue
frontend/src/views/ChatView.vue
frontend/src/components/chat/InputArea.vue
```

Data flow:

```text
ChatView
  -> loads overview once when mounted
  -> passes overview to MessageList
  -> receives suggested question from MessageList
  -> passes it to InputArea as draft text
```

`MessageList` renders `CorpusOverviewCard` only when `messages.length === 0`.

`InputArea` accepts an optional external draft setter. When a suggestion is clicked, the input textarea is filled and focused.

## Empty and Error States

If `/papers/overview` returns zero papers:

- show a compact empty corpus card
- explain that no ingested papers are available
- suggest using the Upload page

If the request fails:

- keep the chat usable
- show a small muted message: `语料概览暂时不可用`
- do not block the input

## Visual Direction

The card should feel like a product overview, not a dashboard:

- centered in the new-chat window
- max width aligned with the chat column
- small neutral metric tiles
- topic rows or compact topic blocks
- representative paper titles are readable but not visually dominant
- no nested cards
- no large marketing hero

The component should match the existing Cite Scope palette and restrained tool UI style.

## Testing

Backend:

- overview returns totals, chunks, and year range
- known bucket categories map to expected labels
- representative papers are deterministic
- empty corpus response is valid

Frontend:

- empty chat renders overview card
- non-empty chat hides overview card
- failed overview request does not break chat
- clicking a suggested question fills the input

Use existing test patterns where available. Do not call external APIs.

## Rollout

Implement in one narrow PR:

1. Backend endpoint and schema.
2. Frontend API/types.
3. `CorpusOverviewCard`.
4. Empty-chat integration.
5. Focused tests and a browser smoke check.

The feature is safe to ship because it is read-only and does not change retrieval or answer generation.
