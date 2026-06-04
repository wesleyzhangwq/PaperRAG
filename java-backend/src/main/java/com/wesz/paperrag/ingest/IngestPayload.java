package com.wesz.paperrag.ingest;

public record IngestPayload(String filename, byte[] bytes) {
}
