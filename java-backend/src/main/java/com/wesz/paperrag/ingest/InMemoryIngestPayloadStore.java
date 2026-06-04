package com.wesz.paperrag.ingest;

import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import org.springframework.stereotype.Component;

@Component
public class InMemoryIngestPayloadStore implements IngestPayloadStore {

    private final ConcurrentMap<Long, IngestPayload> payloads = new ConcurrentHashMap<>();

    @Override
    public void put(Long taskId, IngestPayload payload) {
        payloads.put(taskId, payload);
    }

    @Override
    public Optional<IngestPayload> get(Long taskId) {
        return Optional.ofNullable(payloads.get(taskId));
    }
}
