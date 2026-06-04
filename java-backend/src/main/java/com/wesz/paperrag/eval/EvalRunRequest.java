package com.wesz.paperrag.eval;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import java.util.List;

public record EvalRunRequest(
    @NotBlank String name,
    @NotNull List<Long> groundTruthChunkIds,
    @NotNull List<Long> retrievedChunkIds,
    Integer k
) {

    int resolvedK() {
        return k == null ? 5 : Math.max(1, k);
    }
}
