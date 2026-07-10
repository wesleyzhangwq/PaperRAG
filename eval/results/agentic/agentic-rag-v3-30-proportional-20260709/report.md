# Traditional RAG vs Agentic RAG Eval: agentic-rag-v3-30-proportional-20260709

- Dataset: `questions_v3_200.jsonl`
- Questions: 30
- Selection: {'source_count': 200, 'sample_size': 30, 'per_type': None, 'limit': None}
- Answer metrics are citation/source based and do not use an external judge.

## Key Parameters
| Parameter | Traditional RAG | Agentic RAG |
|---|---:|---:|
| Pipeline | local retrieve -> fixed context answer | LangGraph agent pipeline |
| Retriever | service retriever | planner-routed tools |
| Context strategy | raw | evidence processing + citation gate |
| Context k | 5 | 3 |
| Retrieval top k | None | 12 |
| Max reflections | 0 | 2 |
| Mean steps | 2.0 | 21.5333 |
| Mean retrieval steps | 1.0 | 6.9 |
| Web search configured | false | True |

## Overall Metrics
| Metric | Traditional RAG | Agentic RAG | Delta | Delta % |
|---|---:|---:|---:|---:|
| mode_accuracy | 0.5333 | 0.6333 | 0.1 | 18.75 |
| source_recall | 0.5556 | 0.5556 | 0.0 | 0.0 |
| source_precision | 0.5796 | 0.6164 | 0.0368 | 6.35 |
| citation_support_rate | 1.0 | 0.884 | -0.116 | -11.6 |
| citation_precision | 0.6074 | 0.5855 | -0.0219 | -3.61 |
| citation_expected_hit | 0.6667 | 0.7037 | 0.037 | 5.55 |
| latency_p90 | 12.9855 | 169.5094 | 156.5239 | 1205.37 |
| used_chunks_mean | 4.8333 | 21.1 | 16.2667 | 336.55 |

## Traditional Summary
```json
{
  "count": 30,
  "count_positive": 27,
  "count_negative": 3,
  "mode_accuracy": 0.5333,
  "source_hit": 0.6667,
  "source_recall": 0.5556,
  "source_precision": 0.5796,
  "citation_support_rate": 1.0,
  "citation_precision": 0.6074,
  "citation_expected_hit": 0.6667,
  "used_chunks_mean": 4.8333,
  "step_count_mean": 2.0,
  "retrieval_step_count_mean": 1.0,
  "latency_p50": 6.4021,
  "latency_p90": 12.9855,
  "latency_mean": 8.4276,
  "error_count": 0,
  "by_type": {
    "comparison": {
      "count": 4,
      "count_positive": 4,
      "count_negative": 0,
      "mode_accuracy": 0.25,
      "source_hit": 0.75,
      "source_recall": 0.5,
      "source_precision": 0.3958,
      "citation_support_rate": 1.0,
      "citation_precision": 0.5833,
      "citation_expected_hit": 0.75,
      "used_chunks_mean": 5.0,
      "step_count_mean": 2.0,
      "retrieval_step_count_mean": 1.0,
      "latency_p50": 5.0757,
      "latency_p90": 10.6826,
      "latency_mean": 6.8234,
      "error_count": 0
    },
    "concept_locate": {
      "count": 9,
      "count_positive": 9,
      "count_negative": 0,
      "mode_accuracy": 0.5556,
      "source_hit": 0.5556,
      "source_recall": 0.5556,
      "source_precision": 0.5556,
      "citation_support_rate": 1.0,
      "citation_precision": 0.5556,
      "citation_expected_hit": 0.5556,
      "used_chunks_mean": 5.0,
      "step_count_mean": 2.0,
      "retrieval_step_count_mean": 1.0,
      "latency_p50": 7.9318,
      "latency_p90": 19.9574,
      "latency_mean": 8.7398,
      "error_count": 0
    },
    "fact_extract": {
      "count": 5,
      "count_positive": 5,
      "count_negative": 0,
      "mode_accuracy": 0.8,
      "source_hit": 0.8,
      "source_recall": 0.8,
      "source_precision": 0.8,
      "citation_support_rate": 1.0,
      "citation_precision": 0.8,
      "citation_expected_hit": 0.8,
      "used_chunks_mean": 5.0,
      "step_count_mean": 2.0,
      "retrieval_step_count_mean": 1.0,
      "latency_p50": 4.733,
      "latency_p90": 9.903,
      "latency_mean": 6.3387,
      "error_count": 0
    },
    "method_detail": {
      "count": 6,
      "count_positive": 6,
      "count_negative": 0,
      "mode_accuracy": 0.3333,
      "source_hit": 0.5,
      "source_recall": 0.5,
      "source_precision": 0.5,
      "citation_support_rate": 1.0,
      "citation_precision": 0.5,
      "citation_expected_hit": 0.5,
      "used_chunks_mean": 4.1667,
      "step_count_mean": 2.0,
      "retrieval_step_count_mean": 1.0,
      "latency_p50": 6.4021,
      "latency_p90": 12.9855,
      "latency_mean": 8.0033,
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
      "latency_p50": 4.7826,
      "latency_p90": 5.9102,
      "latency_mean": 4.8303,
      "error_count": 0
    },
    "trend_synthesis": {
      "count": 3,
      "count_positive": 3,
      "count_negative": 0,
      "mode_accuracy": 0.3333,
      "source_hit": 1.0,
      "source_recall": 0.3333,
      "source_precision": 0.6889,
      "citation_support_rate": 1.0,
      "citation_precision": 0.6889,
      "citation_expected_hit": 1.0,
      "used_chunks_mean": 5.0,
      "step_count_mean": 2.0,
      "retrieval_step_count_mean": 1.0,
      "latency_p50": 18.6329,
      "latency_p90": 29.1661,
      "latency_mean": 17.5568,
      "error_count": 0
    }
  },
  "by_difficulty": {
    "easy": {
      "count": 11,
      "count_positive": 9,
      "count_negative": 2,
      "mode_accuracy": 0.6364,
      "source_hit": 0.5556,
      "source_recall": 0.5556,
      "source_precision": 0.5556,
      "citation_support_rate": 1.0,
      "citation_precision": 0.5556,
      "citation_expected_hit": 0.5556,
      "used_chunks_mean": 5.0,
      "step_count_mean": 2.0,
      "retrieval_step_count_mean": 1.0,
      "latency_p50": 7.7848,
      "latency_p90": 9.6848,
      "latency_mean": 8.0333,
      "error_count": 0
    },
    "hard": {
      "count": 7,
      "count_positive": 7,
      "count_negative": 0,
      "mode_accuracy": 0.2857,
      "source_hit": 0.8571,
      "source_recall": 0.4286,
      "source_precision": 0.5214,
      "citation_support_rate": 1.0,
      "citation_precision": 0.6286,
      "citation_expected_hit": 0.8571,
      "used_chunks_mean": 5.0,
      "step_count_mean": 2.0,
      "retrieval_step_count_mean": 1.0,
      "latency_p50": 7.932,
      "latency_p90": 29.1661,
      "latency_mean": 11.4235,
      "error_count": 0
    },
    "medium": {
      "count": 12,
      "count_positive": 11,
      "count_negative": 1,
      "mode_accuracy": 0.5833,
      "source_hit": 0.6364,
      "source_recall": 0.6364,
      "source_precision": 0.6364,
      "citation_support_rate": 1.0,
      "citation_precision": 0.6364,
      "citation_expected_hit": 0.6364,
      "used_chunks_mean": 4.5833,
      "step_count_mean": 2.0,
      "retrieval_step_count_mean": 1.0,
      "latency_p50": 6.2673,
      "latency_p90": 10.8767,
      "latency_mean": 7.0414,
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
  "mode_accuracy": 0.6333,
  "source_hit": 0.6667,
  "source_recall": 0.5556,
  "source_precision": 0.6164,
  "citation_support_rate": 0.884,
  "citation_precision": 0.5855,
  "citation_expected_hit": 0.7037,
  "used_chunks_mean": 21.1,
  "step_count_mean": 21.5333,
  "retrieval_step_count_mean": 6.9,
  "latency_p50": 109.1934,
  "latency_p90": 169.5094,
  "latency_mean": 119.5694,
  "error_count": 0,
  "by_type": {
    "comparison": {
      "count": 4,
      "count_positive": 4,
      "count_negative": 0,
      "mode_accuracy": 0.75,
      "source_hit": 0.75,
      "source_recall": 0.5,
      "source_precision": 0.75,
      "citation_support_rate": 0.8333,
      "citation_precision": 0.625,
      "citation_expected_hit": 0.75,
      "used_chunks_mean": 19.5,
      "step_count_mean": 23.25,
      "retrieval_step_count_mean": 8.75,
      "latency_p50": 91.3306,
      "latency_p90": 151.2512,
      "latency_mean": 103.7858,
      "error_count": 0
    },
    "concept_locate": {
      "count": 9,
      "count_positive": 9,
      "count_negative": 0,
      "mode_accuracy": 0.6667,
      "source_hit": 0.5556,
      "source_recall": 0.5556,
      "source_precision": 0.5,
      "citation_support_rate": 0.9544,
      "citation_precision": 0.537,
      "citation_expected_hit": 0.6667,
      "used_chunks_mean": 20.4444,
      "step_count_mean": 21.6667,
      "retrieval_step_count_mean": 7.6667,
      "latency_p50": 124.5893,
      "latency_p90": 169.5094,
      "latency_mean": 119.4381,
      "error_count": 0
    },
    "fact_extract": {
      "count": 5,
      "count_positive": 5,
      "count_negative": 0,
      "mode_accuracy": 0.6,
      "source_hit": 0.8,
      "source_recall": 0.8,
      "source_precision": 0.8,
      "citation_support_rate": 0.8667,
      "citation_precision": 0.6667,
      "citation_expected_hit": 0.8,
      "used_chunks_mean": 19.2,
      "step_count_mean": 20.4,
      "retrieval_step_count_mean": 5.6,
      "latency_p50": 102.9351,
      "latency_p90": 219.7465,
      "latency_mean": 120.9073,
      "error_count": 0
    },
    "method_detail": {
      "count": 6,
      "count_positive": 6,
      "count_negative": 0,
      "mode_accuracy": 0.3333,
      "source_hit": 0.5,
      "source_recall": 0.5,
      "source_precision": 0.5,
      "citation_support_rate": 0.8,
      "citation_precision": 0.5,
      "citation_expected_hit": 0.5,
      "used_chunks_mean": 23.1667,
      "step_count_mean": 22.6667,
      "retrieval_step_count_mean": 7.3333,
      "latency_p50": 109.1934,
      "latency_p90": 171.9741,
      "latency_mean": 119.5666,
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
      "citation_support_rate": 0.8045,
      "citation_precision": null,
      "citation_expected_hit": null,
      "used_chunks_mean": 23.0,
      "step_count_mean": 18.6667,
      "retrieval_step_count_mean": 4.6667,
      "latency_p50": 103.9092,
      "latency_p90": 123.4733,
      "latency_mean": 101.4444,
      "error_count": 0
    },
    "trend_synthesis": {
      "count": 3,
      "count_positive": 3,
      "count_negative": 0,
      "mode_accuracy": 0.6667,
      "source_hit": 1.0,
      "source_recall": 0.3333,
      "source_precision": 0.7143,
      "citation_support_rate": 1.0,
      "citation_precision": 0.7143,
      "citation_expected_hit": 1.0,
      "used_chunks_mean": 22.3333,
      "step_count_mean": 21.3333,
      "retrieval_step_count_mean": 5.6667,
      "latency_p50": 133.3446,
      "latency_p90": 233.5026,
      "latency_mean": 156.9088,
      "error_count": 0
    }
  },
  "by_difficulty": {
    "easy": {
      "count": 11,
      "count_positive": 9,
      "count_negative": 2,
      "mode_accuracy": 0.7273,
      "source_hit": 0.5556,
      "source_recall": 0.5556,
      "source_precision": 0.5,
      "citation_support_rate": 0.9207,
      "citation_precision": 0.537,
      "citation_expected_hit": 0.6667,
      "used_chunks_mean": 20.7273,
      "step_count_mean": 21.0909,
      "retrieval_step_count_mean": 7.0909,
      "latency_p50": 123.4733,
      "latency_p90": 159.5499,
      "latency_mean": 115.9425,
      "error_count": 0
    },
    "hard": {
      "count": 7,
      "count_positive": 7,
      "count_negative": 0,
      "mode_accuracy": 0.7143,
      "source_hit": 0.8571,
      "source_recall": 0.4286,
      "source_precision": 0.7347,
      "citation_support_rate": 0.9167,
      "citation_precision": 0.6633,
      "citation_expected_hit": 0.8571,
      "used_chunks_mean": 20.7143,
      "step_count_mean": 22.4286,
      "retrieval_step_count_mean": 7.4286,
      "latency_p50": 118.4802,
      "latency_p90": 233.5026,
      "latency_mean": 126.5528,
      "error_count": 0
    },
    "medium": {
      "count": 12,
      "count_positive": 11,
      "count_negative": 1,
      "mode_accuracy": 0.5,
      "source_hit": 0.6364,
      "source_recall": 0.6364,
      "source_precision": 0.6364,
      "citation_support_rate": 0.834,
      "citation_precision": 0.5758,
      "citation_expected_hit": 0.6364,
      "used_chunks_mean": 21.6667,
      "step_count_mean": 21.4167,
      "retrieval_step_count_mean": 6.4167,
      "latency_p50": 103.9092,
      "latency_p90": 171.9741,
      "latency_mean": 118.8204,
      "error_count": 0
    }
  }
}
```

## Repair
- Retried failed rows at 2026-07-09T18:51:53.274860+00:00 (max attempts: 1).
- Remaining errors: traditional=0, agentic=0.