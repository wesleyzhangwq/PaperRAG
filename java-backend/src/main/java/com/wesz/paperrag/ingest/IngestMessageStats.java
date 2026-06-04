package com.wesz.paperrag.ingest;

public record IngestMessageStats(
    Long lastTaskId,
    long publishedCount,
    long consumedCount,
    long failedCount
) {
}
