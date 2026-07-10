# Traditional RAG vs Agentic RAG Eval: agentic-rag-v5-30-local-bge-m3-20260710

- Dataset: `questions_v3_200.jsonl`
- Questions: 30
- Selection: {'source_count': 200, 'sample_size': 30, 'per_type': None, 'limit': None}
- Retrieval metrics use raw local paper IDs; citation support uses final context paper IDs.
- Answer metrics do not use an external judge.

## Key Parameters
| Parameter | Traditional RAG | Agentic RAG |
|---|---:|---:|
| Pipeline | local retrieve -> fixed context answer | LangGraph agent pipeline |
| Retriever | service retriever | planner-routed tools |
| Context strategy | raw | evidence processing + citation gate |
| Context k | 5 | 3 |
| Retrieval top k | 12 | 12 |
| Max reflections | 0 | 2 |
| Mean steps | 2.0 | 22.0 |
| Mean retrieval steps | 1.0 | 7.2667 |
| External retrieval | disabled | disabled |

## Overall Metrics
| Metric | Traditional RAG | Agentic RAG | Delta | Delta % |
|---|---:|---:|---:|---:|
| mode_accuracy | 0.7333 | 0.7333 | 0.0 | 0.0 |
| source_recall | 0.8667 | 0.8963 | 0.0296 | 3.42 |
| source_precision | 0.4587 | 0.5355 | 0.0768 | 16.74 |
| citation_support_rate | 0.9655 | 0.9828 | 0.0173 | 1.79 |
| citation_precision | 0.7926 | 0.7778 | -0.0148 | -1.87 |
| citation_expected_hit | 0.8889 | 0.8889 | 0.0 | 0.0 |
| latency_p90 | 20.0643 | 221.481 | 201.4167 | 1003.86 |
| used_chunks_mean | 5.0 | 3.0 | -2.0 | -40.0 |

## Traditional Summary
```json
{
  "count": 30,
  "count_positive": 27,
  "count_negative": 3,
  "mode_accuracy": 0.7333,
  "source_hit": 0.963,
  "source_recall": 0.8667,
  "source_precision": 0.4587,
  "citation_support_rate": 0.9655,
  "citation_precision": 0.7926,
  "citation_expected_hit": 0.8889,
  "used_chunks_mean": 5.0,
  "step_count_mean": 2.0,
  "retrieval_step_count_mean": 1.0,
  "latency_p50": 9.8087,
  "latency_p90": 20.0643,
  "latency_mean": 12.6334,
  "error_count": 0,
  "by_type": {
    "comparison": {
      "count": 4,
      "count_positive": 4,
      "count_negative": 0,
      "mode_accuracy": 0.5,
      "source_hit": 1.0,
      "source_recall": 0.75,
      "source_precision": 0.2977,
      "citation_support_rate": 1.0,
      "citation_precision": 0.875,
      "citation_expected_hit": 1.0,
      "used_chunks_mean": 5.0,
      "step_count_mean": 2.0,
      "retrieval_step_count_mean": 1.0,
      "latency_p50": 10.1266,
      "latency_p90": 26.1098,
      "latency_mean": 14.108,
      "error_count": 0
    },
    "concept_locate": {
      "count": 9,
      "count_positive": 9,
      "count_negative": 0,
      "mode_accuracy": 0.8889,
      "source_hit": 1.0,
      "source_recall": 1.0,
      "source_precision": 0.4,
      "citation_support_rate": 0.8889,
      "citation_precision": 0.8889,
      "citation_expected_hit": 1.0,
      "used_chunks_mean": 5.0,
      "step_count_mean": 2.0,
      "retrieval_step_count_mean": 1.0,
      "latency_p50": 10.416,
      "latency_p90": 15.9216,
      "latency_mean": 11.3633,
      "error_count": 0
    },
    "fact_extract": {
      "count": 5,
      "count_positive": 5,
      "count_negative": 0,
      "mode_accuracy": 0.8,
      "source_hit": 0.8,
      "source_recall": 0.8,
      "source_precision": 0.7,
      "citation_support_rate": 1.0,
      "citation_precision": 0.6,
      "citation_expected_hit": 0.6,
      "used_chunks_mean": 5.0,
      "step_count_mean": 2.0,
      "retrieval_step_count_mean": 1.0,
      "latency_p50": 7.3355,
      "latency_p90": 27.5936,
      "latency_mean": 11.3549,
      "error_count": 0
    },
    "method_detail": {
      "count": 6,
      "count_positive": 6,
      "count_negative": 0,
      "mode_accuracy": 0.6667,
      "source_hit": 1.0,
      "source_recall": 1.0,
      "source_precision": 0.503,
      "citation_support_rate": 1.0,
      "citation_precision": 1.0,
      "citation_expected_hit": 1.0,
      "used_chunks_mean": 5.0,
      "step_count_mean": 2.0,
      "retrieval_step_count_mean": 1.0,
      "latency_p50": 9.8087,
      "latency_p90": 34.1042,
      "latency_mean": 14.3835,
      "error_count": 0
    },
    "negative": {
      "count": 3,
      "count_positive": 0,
      "count_negative": 3,
      "mode_accuracy": 1.0,
      "source_hit": null,
      "source_recall": null,
      "source_precision": null,
      "citation_support_rate": 1.0,
      "citation_precision": null,
      "citation_expected_hit": null,
      "used_chunks_mean": 5.0,
      "step_count_mean": 2.0,
      "retrieval_step_count_mean": 1.0,
      "latency_p50": 7.9765,
      "latency_p90": 8.6518,
      "latency_mean": 7.7934,
      "error_count": 0
    },
    "trend_synthesis": {
      "count": 3,
      "count_positive": 3,
      "count_negative": 0,
      "mode_accuracy": 0.3333,
      "source_hit": 1.0,
      "source_recall": 0.4667,
      "source_precision": 0.3583,
      "citation_support_rate": 1.0,
      "citation_precision": 0.3,
      "citation_expected_hit": 0.6667,
      "used_chunks_mean": 5.0,
      "step_count_mean": 2.0,
      "retrieval_step_count_mean": 1.0,
      "latency_p50": 17.2219,
      "latency_p90": 20.0643,
      "latency_mean": 17.9482,
      "error_count": 0
    }
  },
  "by_difficulty": {
    "easy": {
      "count": 11,
      "count_positive": 9,
      "count_negative": 2,
      "mode_accuracy": 0.9091,
      "source_hit": 1.0,
      "source_recall": 1.0,
      "source_precision": 0.4,
      "citation_support_rate": 0.9091,
      "citation_precision": 0.8889,
      "citation_expected_hit": 1.0,
      "used_chunks_mean": 5.0,
      "step_count_mean": 2.0,
      "retrieval_step_count_mean": 1.0,
      "latency_p50": 9.5716,
      "latency_p90": 15.2063,
      "latency_mean": 10.6362,
      "error_count": 0
    },
    "hard": {
      "count": 7,
      "count_positive": 7,
      "count_negative": 0,
      "mode_accuracy": 0.4286,
      "source_hit": 1.0,
      "source_recall": 0.6286,
      "source_precision": 0.3237,
      "citation_support_rate": 1.0,
      "citation_precision": 0.6286,
      "citation_expected_hit": 0.8571,
      "used_chunks_mean": 5.0,
      "step_count_mean": 2.0,
      "retrieval_step_count_mean": 1.0,
      "latency_p50": 16.5583,
      "latency_p90": 26.1098,
      "latency_mean": 15.7538,
      "error_count": 0
    },
    "medium": {
      "count": 12,
      "count_positive": 11,
      "count_negative": 1,
      "mode_accuracy": 0.75,
      "source_hit": 0.9091,
      "source_recall": 0.9091,
      "source_precision": 0.5925,
      "citation_support_rate": 1.0,
      "citation_precision": 0.8182,
      "citation_expected_hit": 0.8182,
      "used_chunks_mean": 5.0,
      "step_count_mean": 2.0,
      "retrieval_step_count_mean": 1.0,
      "latency_p50": 8.9519,
      "latency_p90": 27.5936,
      "latency_mean": 12.644,
      "error_count": 0
    }
  }
}
```

## Agentic Summary
```json
{
  "count": 30,
  "count_positive": 27,
  "count_negative": 3,
  "mode_accuracy": 0.7333,
  "source_hit": 0.963,
  "source_recall": 0.8963,
  "source_precision": 0.5355,
  "citation_support_rate": 0.9828,
  "citation_precision": 0.7778,
  "citation_expected_hit": 0.8889,
  "used_chunks_mean": 3.0,
  "step_count_mean": 22.0,
  "retrieval_step_count_mean": 7.2667,
  "latency_p50": 138.887,
  "latency_p90": 221.481,
  "latency_mean": 178.033,
  "error_count": 0,
  "by_type": {
    "comparison": {
      "count": 4,
      "count_positive": 4,
      "count_negative": 0,
      "mode_accuracy": 0.25,
      "source_hit": 1.0,
      "source_recall": 1.0,
      "source_precision": 0.5881,
      "citation_support_rate": 0.875,
      "citation_precision": 0.5833,
      "citation_expected_hit": 1.0,
      "used_chunks_mean": 3.0,
      "step_count_mean": 28.0,
      "retrieval_step_count_mean": 10.0,
      "latency_p50": 209.4531,
      "latency_p90": 715.5297,
      "latency_mean": 380.6452,
      "error_count": 0
    },
    "concept_locate": {
      "count": 9,
      "count_positive": 9,
      "count_negative": 0,
      "mode_accuracy": 0.8889,
      "source_hit": 1.0,
      "source_recall": 1.0,
      "source_precision": 0.5457,
      "citation_support_rate": 1.0,
      "citation_precision": 0.9259,
      "citation_expected_hit": 1.0,
      "used_chunks_mean": 3.0,
      "step_count_mean": 20.3333,
      "retrieval_step_count_mean": 7.1111,
      "latency_p50": 127.0858,
      "latency_p90": 214.1213,
      "latency_mean": 127.9739,
      "error_count": 0
    },
    "fact_extract": {
      "count": 5,
      "count_positive": 5,
      "count_negative": 0,
      "mode_accuracy": 1.0,
      "source_hit": 0.8,
      "source_recall": 0.8,
      "source_precision": 0.6182,
      "citation_support_rate": 1.0,
      "citation_precision": 0.6,
      "citation_expected_hit": 0.6,
      "used_chunks_mean": 3.0,
      "step_count_mean": 19.8,
      "retrieval_step_count_mean": 5.8,
      "latency_p50": 129.1662,
      "latency_p90": 177.1112,
      "latency_mean": 124.3425,
      "error_count": 0
    },
    "method_detail": {
      "count": 6,
      "count_positive": 6,
      "count_negative": 0,
      "mode_accuracy": 0.8333,
      "source_hit": 1.0,
      "source_recall": 1.0,
      "source_precision": 0.5583,
      "citation_support_rate": 1.0,
      "citation_precision": 0.8889,
      "citation_expected_hit": 1.0,
      "used_chunks_mean": 3.0,
      "step_count_mean": 22.5,
      "retrieval_step_count_mean": 7.8333,
      "latency_p50": 151.4767,
      "latency_p90": 369.6057,
      "latency_mean": 183.7228,
      "error_count": 0
    },
    "negative": {
      "count": 3,
      "count_positive": 0,
      "count_negative": 3,
      "mode_accuracy": 1.0,
      "source_hit": null,
      "source_recall": null,
      "source_precision": null,
      "citation_support_rate": 1.0,
      "citation_precision": null,
      "citation_expected_hit": null,
      "used_chunks_mean": 3.0,
      "step_count_mean": 21.3333,
      "retrieval_step_count_mean": 6.0,
      "latency_p50": 128.7983,
      "latency_p90": 138.887,
      "latency_mean": 126.9806,
      "error_count": 0
    },
    "trend_synthesis": {
      "count": 3,
      "count_positive": 3,
      "count_negative": 0,
      "mode_accuracy": 0.0,
      "source_hit": 1.0,
      "source_recall": 0.4,
      "source_precision": 0.2518,
      "citation_support_rate": 1.0,
      "citation_precision": 0.6667,
      "citation_expected_hit": 0.6667,
      "used_chunks_mean": 3.0,
      "step_count_mean": 22.3333,
      "retrieval_step_count_mean": 6.6667,
      "latency_p50": 187.6135,
      "latency_p90": 211.031,
      "latency_mean": 187.2178,
      "error_count": 0
    }
  },
  "by_difficulty": {
    "easy": {
      "count": 11,
      "count_positive": 9,
      "count_negative": 2,
      "mode_accuracy": 0.9091,
      "source_hit": 1.0,
      "source_recall": 1.0,
      "source_precision": 0.5457,
      "citation_support_rate": 1.0,
      "citation_precision": 0.9259,
      "citation_expected_hit": 1.0,
      "used_chunks_mean": 3.0,
      "step_count_mean": 20.8182,
      "retrieval_step_count_mean": 7.0909,
      "latency_p50": 127.0858,
      "latency_p90": 152.964,
      "latency_mean": 126.7109,
      "error_count": 0
    },
    "hard": {
      "count": 7,
      "count_positive": 7,
      "count_negative": 0,
      "mode_accuracy": 0.1429,
      "source_hit": 1.0,
      "source_recall": 0.7429,
      "source_precision": 0.4439,
      "citation_support_rate": 0.9286,
      "citation_precision": 0.619,
      "citation_expected_hit": 0.8571,
      "used_chunks_mean": 3.0,
      "step_count_mean": 25.5714,
      "retrieval_step_count_mean": 8.5714,
      "latency_p50": 209.4531,
      "latency_p90": 715.5297,
      "latency_mean": 297.7477,
      "error_count": 0
    },
    "medium": {
      "count": 12,
      "count_positive": 11,
      "count_negative": 1,
      "mode_accuracy": 0.9167,
      "source_hit": 0.9091,
      "source_recall": 0.9091,
      "source_precision": 0.5855,
      "citation_support_rate": 1.0,
      "citation_precision": 0.7576,
      "citation_expected_hit": 0.8182,
      "used_chunks_mean": 3.0,
      "step_count_mean": 21.0,
      "retrieval_step_count_mean": 6.6667,
      "latency_p50": 138.887,
      "latency_p90": 221.481,
      "latency_mean": 155.2447,
      "error_count": 0
    }
  }
}
```