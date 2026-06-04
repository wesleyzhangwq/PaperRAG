package com.wesz.paperrag.chat;

import com.wesz.paperrag.vector.RetrievalResultResponse;

public record SourceResponse(
    Long chunkId,
    Long paperId,
    Integer chunkIndex,
    String content,
    double score
) {

    public static SourceResponse from(RetrievalResultResponse result) {
        return new SourceResponse(
            result.chunkId(),
            result.paperId(),
            result.chunkIndex(),
            result.content(),
            result.score()
        );
    }
}
