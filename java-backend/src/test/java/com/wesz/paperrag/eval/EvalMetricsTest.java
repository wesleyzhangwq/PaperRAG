package com.wesz.paperrag.eval;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;
import org.junit.jupiter.api.Test;

class EvalMetricsTest {

    @Test
    void calculatesRecallMrrAndNdcgAtK() {
        EvalMetrics metrics = EvalMetrics.calculate(
            List.of(101L, 103L),
            List.of(101L, 102L, 103L),
            2
        );

        assertThat(metrics.recallAtK()).isEqualTo(0.5);
        assertThat(metrics.mrr()).isEqualTo(1.0);
        assertThat(metrics.ndcgAtK()).isEqualTo(0.6131);
    }

    @Test
    void handlesNoRelevantDocuments() {
        EvalMetrics metrics = EvalMetrics.calculate(List.of(), List.of(1L, 2L), 2);

        assertThat(metrics.recallAtK()).isZero();
        assertThat(metrics.mrr()).isZero();
        assertThat(metrics.ndcgAtK()).isZero();
    }
}
