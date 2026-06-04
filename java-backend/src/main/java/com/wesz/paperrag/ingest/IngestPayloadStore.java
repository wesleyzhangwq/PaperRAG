package com.wesz.paperrag.ingest;

import java.util.Optional;

public interface IngestPayloadStore {

    void put(Long taskId, IngestPayload payload);

    Optional<IngestPayload> get(Long taskId);
}
