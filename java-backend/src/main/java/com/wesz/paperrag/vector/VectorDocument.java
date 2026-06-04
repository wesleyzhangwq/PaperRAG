package com.wesz.paperrag.vector;

public record VectorDocument(
    Long chunkId,
    Long paperId,
    Integer chunkIndex,
    String content,
    float[] vector
) {
}
