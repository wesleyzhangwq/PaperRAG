package com.wesz.paperrag.ingest;

public record IngestUploadResponse(Long paperId, Long taskId, IngestStatus status) {
}
