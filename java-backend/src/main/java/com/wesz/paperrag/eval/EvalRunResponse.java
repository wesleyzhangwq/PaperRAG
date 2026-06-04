package com.wesz.paperrag.eval;

import java.time.Instant;

public record EvalRunResponse(
    Long runId,
    String name,
    int k,
    double recallAtK,
    double mrr,
    double ndcgAtK,
    Instant createdAt
) {

    static EvalRunResponse from(EvalRun run) {
        return new EvalRunResponse(
            run.getId(),
            run.getName(),
            run.getK(),
            run.getRecallAtK(),
            run.getMrr(),
            run.getNdcgAtK(),
            run.getCreatedAt()
        );
    }
}
