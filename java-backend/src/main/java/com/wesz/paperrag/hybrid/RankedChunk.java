package com.wesz.paperrag.hybrid;

record RankedChunk(
    Long chunkId,
    Long paperId,
    Integer chunkIndex,
    String content,
    double score,
    String channel
) {
}
