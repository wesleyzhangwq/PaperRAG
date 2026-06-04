package com.wesz.paperrag.ingest;

import org.springframework.stereotype.Component;

@Component
public class IngestMessageConsumer {

    private final IngestWorker ingestWorker;

    public IngestMessageConsumer(IngestWorker ingestWorker) {
        this.ingestWorker = ingestWorker;
    }

    public IngestStatus consume(IngestMessage message) {
        return ingestWorker.process(
            message.taskId(),
            message.paperId(),
            message.tenantId(),
            message.filename(),
            message.bytes()
        );
    }
}
