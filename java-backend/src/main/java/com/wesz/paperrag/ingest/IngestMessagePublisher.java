package com.wesz.paperrag.ingest;

public interface IngestMessagePublisher {

    void publish(IngestMessage message);

    IngestMessageStats stats();
}
