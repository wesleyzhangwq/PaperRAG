package com.wesz.paperrag.ingest;

public record IngestMessage(
    Long taskId,
    Long paperId,
    Long tenantId,
    String filename,
    byte[] bytes,
    int attempt
) {
}
