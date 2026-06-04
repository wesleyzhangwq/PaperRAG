package com.wesz.paperrag.ingest;

public record IngestRetryResponse(Long taskId, IngestStatus status, boolean republished) {
}
