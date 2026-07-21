# Traditional RAG vs Agentic RAG Eval: agentic-rag-501-v2-30-local-tuned

- Dataset: `questions_501_test_200.jsonl`
- Questions: 30
- Selection: {'source_count': 200, 'sample_size': 30, 'per_type': None, 'limit': None}
- Retrieval metrics use raw local paper IDs; citation support uses final context paper IDs.
- Answer metrics do not use an external judge.

## Key Parameters
| Parameter | Traditional RAG | Agentic RAG |
|---|---:|---:|
| Pipeline | local retrieve -> fixed context answer | LangGraph agent pipeline |
| Retriever | service retriever | planner-routed tools |
| Context strategy | paper_dedup | evidence processing + citation gate |
| Context k | 5 | 5 |
| Retrieval top k | 20 | 20 |
| Max reflections | 0 | 2 |
| Mean steps | 2.0 | 18.2667 |
| Mean retrieval steps | 1.0 | 3.7333 |
| External retrieval | disabled | disabled |

## Overall Metrics
| Metric | Traditional RAG | Agentic RAG | Delta | Delta % |
|---|---:|---:|---:|---:|
| mode_accuracy | 0.7333 | 0.9333 | 0.2 | 27.27 |
| source_recall | 0.8074 | 0.8704 | 0.063 | 7.8 |
| source_precision | 0.1472 | 0.2479 | 0.1007 | 68.41 |
| citation_support_rate | 0.9833 | 1.0 | 0.0167 | 1.7 |
| citation_precision | 0.5969 | 0.6012 | 0.0043 | 0.72 |
| citation_expected_hit | 0.8889 | 0.9259 | 0.037 | 4.16 |
| latency_p90 | 15.912 | 140.9424 | 125.0304 | 785.76 |
| used_chunks_mean | 4.6667 | 4.9333 | 0.2666 | 5.71 |

## Paired Bootstrap (10,000 samples)

| Metric | Traditional RAG | Agentic RAG | Mean delta | 95% CI | W/T/L |
|---|---:|---:|---:|---:|---:|
| answer/refusal mode correct | 0.7333 | 0.9333 | +0.2000 | [+0.0333, +0.3667] | 7/22/1 |
| raw retrieval source recall | 0.8074 | 0.8704 | +0.0630 | [+0.0000, +0.1556] | 3/24/0 |
| raw retrieval source precision | 0.1472 | 0.2479 | +0.1007 | [+0.0150, +0.1922] | 20/1/6 |
| final-context citation support | 0.9833 | 1.0000 | +0.0167 | [+0.0000, +0.0500] | 1/29/0 |
| expected-paper citation precision | 0.5969 | 0.6012 | +0.0043 | [-0.1136, +0.1191] | 8/14/5 |
| end-to-end latency (mean seconds) | 9.7493 | 98.0507 | +88.3014 | [+77.0742, +99.9522] | 0/0/30 |

`answer/refusal mode correct` only checks whether a positive question receives a substantive answer and a negative question receives an overall abstention. Partial caveats inside an otherwise substantive answer are not counted as full abstention. No external answer-quality judge is used. Full paired output is in `paired-bootstrap/`.

## Traditional Summary
```json
{
  "count": 30,
  "count_positive": 27,
  "count_negative": 3,
  "mode_accuracy": 0.7333,
  "source_hit": 0.9259,
  "source_recall": 0.8074,
  "source_precision": 0.1472,
  "citation_support_rate": 0.9833,
  "citation_precision": 0.5969,
  "citation_expected_hit": 0.8889,
  "used_chunks_mean": 4.6667,
  "step_count_mean": 2.0,
  "retrieval_step_count_mean": 1.0,
  "latency_p50": 6.7687,
  "latency_p90": 15.912,
  "latency_mean": 9.7493,
  "error_count": 0,
  "by_type": {
    "comparison": {
      "count": 4,
      "count_positive": 4,
      "count_negative": 0,
      "mode_accuracy": 0.5,
      "source_hit": 0.75,
      "source_recall": 0.5,
      "source_precision": 0.1503,
      "citation_support_rate": 1.0,
      "citation_precision": 0.625,
      "citation_expected_hit": 0.75,
      "used_chunks_mean": 4.5,
      "step_count_mean": 2.0,
      "retrieval_step_count_mean": 1.0,
      "latency_p50": 6.379,
      "latency_p90": 15.6038,
      "latency_mean": 9.8478,
      "error_count": 0
    },
    "concept_locate": {
      "count": 9,
      "count_positive": 9,
      "count_negative": 0,
      "mode_accuracy": 0.8889,
      "source_hit": 1.0,
      "source_recall": 1.0,
      "source_precision": 0.1596,
      "citation_support_rate": 0.9444,
      "citation_precision": 0.4796,
      "citation_expected_hit": 1.0,
      "used_chunks_mean": 4.4444,
      "step_count_mean": 2.0,
      "retrieval_step_count_mean": 1.0,
      "latency_p50": 11.2711,
      "latency_p90": 15.0979,
      "latency_mean": 10.333,
      "error_count": 0
    },
    "fact_extract": {
      "count": 5,
      "count_positive": 5,
      "count_negative": 0,
      "mode_accuracy": 0.4,
      "source_hit": 1.0,
      "source_recall": 1.0,
      "source_precision": 0.1358,
      "citation_support_rate": 1.0,
      "citation_precision": 0.74,
      "citation_expected_hit": 1.0,
      "used_chunks_mean": 5.0,
      "step_count_mean": 2.0,
      "retrieval_step_count_mean": 1.0,
      "latency_p50": 6.4086,
      "latency_p90": 6.7892,
      "latency_mean": 5.6528,
      "error_count": 0
    },
    "method_detail": {
      "count": 6,
      "count_positive": 6,
      "count_negative": 0,
      "mode_accuracy": 0.6667,
      "source_hit": 0.8333,
      "source_recall": 0.8333,
      "source_precision": 0.1587,
      "citation_support_rate": 1.0,
      "citation_precision": 0.8333,
      "citation_expected_hit": 0.8333,
      "used_chunks_mean": 4.5,
      "step_count_mean": 2.0,
      "retrieval_step_count_mean": 1.0,
      "latency_p50": 6.7349,
      "latency_p90": 15.912,
      "latency_mean": 8.3727,
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
      "latency_p50": 3.9952,
      "latency_p90": 6.5522,
      "latency_mean": 4.705,
      "error_count": 0
    },
    "trend_synthesis": {
      "count": 3,
      "count_positive": 3,
      "count_negative": 0,
      "mode_accuracy": 1.0,
      "source_hit": 1.0,
      "source_recall": 0.2667,
      "source_precision": 0.1016,
      "citation_support_rate": 1.0,
      "citation_precision": 0.2,
      "citation_expected_hit": 0.6667,
      "used_chunks_mean": 5.0,
      "step_count_mean": 2.0,
      "retrieval_step_count_mean": 1.0,
      "latency_p50": 23.2146,
      "latency_p90": 27.2519,
      "latency_mean": 22.4918,
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
      "source_precision": 0.1596,
      "citation_support_rate": 0.9545,
      "citation_precision": 0.4796,
      "citation_expected_hit": 1.0,
      "used_chunks_mean": 4.5455,
      "step_count_mean": 2.0,
      "retrieval_step_count_mean": 1.0,
      "latency_p50": 8.9844,
      "latency_p90": 13.5579,
      "latency_mean": 9.4131,
      "error_count": 0
    },
    "hard": {
      "count": 7,
      "count_positive": 7,
      "count_negative": 0,
      "mode_accuracy": 0.7143,
      "source_hit": 0.8571,
      "source_recall": 0.4,
      "source_precision": 0.1294,
      "citation_support_rate": 1.0,
      "citation_precision": 0.4429,
      "citation_expected_hit": 0.7143,
      "used_chunks_mean": 4.7143,
      "step_count_mean": 2.0,
      "retrieval_step_count_mean": 1.0,
      "latency_p50": 15.6038,
      "latency_p90": 27.2519,
      "latency_mean": 15.2667,
      "error_count": 0
    },
    "medium": {
      "count": 12,
      "count_positive": 11,
      "count_negative": 1,
      "mode_accuracy": 0.5833,
      "source_hit": 0.9091,
      "source_recall": 0.9091,
      "source_precision": 0.1483,
      "citation_support_rate": 1.0,
      "citation_precision": 0.7909,
      "citation_expected_hit": 0.9091,
      "used_chunks_mean": 4.75,
      "step_count_mean": 2.0,
      "retrieval_step_count_mean": 1.0,
      "latency_p50": 6.4086,
      "latency_p90": 9.2086,
      "latency_mean": 6.839,
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
  "mode_accuracy": 0.9333,
  "source_hit": 0.963,
  "source_recall": 0.8704,
  "source_precision": 0.2479,
  "citation_support_rate": 1.0,
  "citation_precision": 0.6012,
  "citation_expected_hit": 0.9259,
  "used_chunks_mean": 4.9333,
  "step_count_mean": 18.2667,
  "retrieval_step_count_mean": 3.7333,
  "latency_p50": 98.8338,
  "latency_p90": 140.9424,
  "latency_mean": 98.0507,
  "error_count": 0,
  "by_type": {
    "comparison": {
      "count": 4,
      "count_positive": 4,
      "count_negative": 0,
      "mode_accuracy": 1.0,
      "source_hit": 1.0,
      "source_recall": 0.875,
      "source_precision": 0.2444,
      "citation_support_rate": 1.0,
      "citation_precision": 0.55,
      "citation_expected_hit": 1.0,
      "used_chunks_mean": 5.0,
      "step_count_mean": 20.0,
      "retrieval_step_count_mean": 5.75,
      "latency_p50": 87.3553,
      "latency_p90": 150.1055,
      "latency_mean": 98.7229,
      "error_count": 0
    },
    "concept_locate": {
      "count": 9,
      "count_positive": 9,
      "count_negative": 0,
      "mode_accuracy": 1.0,
      "source_hit": 1.0,
      "source_recall": 1.0,
      "source_precision": 0.2662,
      "citation_support_rate": 1.0,
      "citation_precision": 0.6481,
      "citation_expected_hit": 1.0,
      "used_chunks_mean": 4.8889,
      "step_count_mean": 17.8889,
      "retrieval_step_count_mean": 3.2222,
      "latency_p50": 98.8338,
      "latency_p90": 140.9424,
      "latency_mean": 99.8174,
      "error_count": 0
    },
    "fact_extract": {
      "count": 5,
      "count_positive": 5,
      "count_negative": 0,
      "mode_accuracy": 1.0,
      "source_hit": 1.0,
      "source_recall": 1.0,
      "source_precision": 0.2643,
      "citation_support_rate": 1.0,
      "citation_precision": 0.8,
      "citation_expected_hit": 1.0,
      "used_chunks_mean": 5.0,
      "step_count_mean": 18.0,
      "retrieval_step_count_mean": 3.4,
      "latency_p50": 86.765,
      "latency_p90": 101.9654,
      "latency_mean": 77.8873,
      "error_count": 0
    },
    "method_detail": {
      "count": 6,
      "count_positive": 6,
      "count_negative": 0,
      "mode_accuracy": 0.8333,
      "source_hit": 0.8333,
      "source_recall": 0.8333,
      "source_precision": 0.267,
      "citation_support_rate": 1.0,
      "citation_precision": 0.625,
      "citation_expected_hit": 0.8333,
      "used_chunks_mean": 4.8333,
      "step_count_mean": 16.8333,
      "retrieval_step_count_mean": 3.3333,
      "latency_p50": 101.0734,
      "latency_p90": 128.0698,
      "latency_mean": 92.2741,
      "error_count": 0
    },
    "negative": {
      "count": 3,
      "count_positive": 0,
      "count_negative": 3,
      "mode_accuracy": 0.6667,
      "source_hit": null,
      "source_recall": null,
      "source_precision": null,
      "citation_support_rate": 1.0,
      "citation_precision": null,
      "citation_expected_hit": null,
      "used_chunks_mean": 5.0,
      "step_count_mean": 18.3333,
      "retrieval_step_count_mean": 3.0,
      "latency_p50": 81.2166,
      "latency_p90": 106.0795,
      "latency_mean": 86.5846,
      "error_count": 0
    },
    "trend_synthesis": {
      "count": 3,
      "count_positive": 3,
      "count_negative": 0,
      "mode_accuracy": 1.0,
      "source_hit": 1.0,
      "source_recall": 0.3333,
      "source_precision": 0.1319,
      "citation_support_rate": 1.0,
      "citation_precision": 0.15,
      "citation_expected_hit": 0.6667,
      "used_chunks_mean": 5.0,
      "step_count_mean": 20.3333,
      "retrieval_step_count_mean": 4.6667,
      "latency_p50": 156.3778,
      "latency_p90": 182.0833,
      "latency_mean": 148.479,
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
      "source_precision": 0.2662,
      "citation_support_rate": 1.0,
      "citation_precision": 0.6481,
      "citation_expected_hit": 1.0,
      "used_chunks_mean": 4.9091,
      "step_count_mean": 17.8182,
      "retrieval_step_count_mean": 3.1818,
      "latency_p50": 81.2166,
      "latency_p90": 138.9056,
      "latency_mean": 95.6392,
      "error_count": 0
    },
    "hard": {
      "count": 7,
      "count_positive": 7,
      "count_negative": 0,
      "mode_accuracy": 1.0,
      "source_hit": 1.0,
      "source_recall": 0.6429,
      "source_precision": 0.1962,
      "citation_support_rate": 1.0,
      "citation_precision": 0.3786,
      "citation_expected_hit": 0.8571,
      "used_chunks_mean": 5.0,
      "step_count_mean": 20.1429,
      "retrieval_step_count_mean": 5.2857,
      "latency_p50": 106.976,
      "latency_p90": 182.0833,
      "latency_mean": 120.047,
      "error_count": 0
    },
    "medium": {
      "count": 12,
      "count_positive": 11,
      "count_negative": 1,
      "mode_accuracy": 0.9167,
      "source_hit": 0.9091,
      "source_recall": 0.9091,
      "source_precision": 0.2658,
      "citation_support_rate": 1.0,
      "citation_precision": 0.7045,
      "citation_expected_hit": 0.9091,
      "used_chunks_mean": 4.9167,
      "step_count_mean": 17.5833,
      "retrieval_step_count_mean": 3.3333,
      "latency_p50": 99.7523,
      "latency_p90": 118.9428,
      "latency_mean": 87.43,
      "error_count": 0
    }
  }
}
```

## Repair
- Retried failed rows at 2026-07-11T13:45:35.657969+00:00 (max attempts: 1).
- Remaining errors: traditional=0, agentic=0.
