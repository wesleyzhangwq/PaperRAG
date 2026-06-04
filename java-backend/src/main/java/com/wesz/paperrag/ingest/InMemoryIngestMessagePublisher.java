package com.wesz.paperrag.ingest;

import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import org.springframework.stereotype.Component;

@Component
public class InMemoryIngestMessagePublisher implements IngestMessagePublisher {

    private final IngestMessageConsumer consumer;
    private final AtomicReference<Long> lastTaskId = new AtomicReference<>();
    private final AtomicLong publishedCount = new AtomicLong();
    private final AtomicLong consumedCount = new AtomicLong();
    private final AtomicLong failedCount = new AtomicLong();

    public InMemoryIngestMessagePublisher(IngestMessageConsumer consumer) {
        this.consumer = consumer;
    }

    @Override
    public void publish(IngestMessage message) {
        lastTaskId.set(message.taskId());
        publishedCount.incrementAndGet();
        IngestStatus status = consumer.consume(message);
        if (status == IngestStatus.FAILED) {
            failedCount.incrementAndGet();
        } else {
            consumedCount.incrementAndGet();
        }
    }

    @Override
    public IngestMessageStats stats() {
        return new IngestMessageStats(
            lastTaskId.get(),
            publishedCount.get(),
            consumedCount.get(),
            failedCount.get()
        );
    }
}
