package com.wesz.paperrag.vector;

public record VectorSearchResult(
    Long chunkId,
    Long paperId,
    Integer chunkIndex,
    String content,
    double score
) {
}
