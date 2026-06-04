package com.wesz.paperrag.vector;

public record RetrievalResultResponse(
    Long chunkId,
    Long paperId,
    Integer chunkIndex,
    String content,
    double score
) {

    static RetrievalResultResponse from(VectorSearchResult result) {
        return new RetrievalResultResponse(
            result.chunkId(),
            result.paperId(),
            result.chunkIndex(),
            result.content(),
            result.score()
        );
    }
}
