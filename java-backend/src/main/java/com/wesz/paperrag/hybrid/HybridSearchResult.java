package com.wesz.paperrag.hybrid;

import java.util.List;

public record HybridSearchResult(
    Long chunkId,
    Long paperId,
    Integer chunkIndex,
    String content,
    double score,
    List<String> channels
) {
}
