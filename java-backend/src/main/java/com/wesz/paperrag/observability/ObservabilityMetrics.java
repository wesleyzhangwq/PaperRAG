package com.wesz.paperrag.observability;

import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;
import java.util.function.Supplier;
import org.springframework.stereotype.Component;

@Component
public class ObservabilityMetrics {

    private final Counter chatRequests;
    private final Timer retrievalSearchTimer;
    private final MeterRegistry meterRegistry;

    public ObservabilityMetrics(MeterRegistry meterRegistry) {
        this.meterRegistry = meterRegistry;
        this.chatRequests = Counter.builder("paperrag.chat.requests")
            .description("Total PaperRAG chat requests")
            .register(meterRegistry);
        this.retrievalSearchTimer = Timer.builder("paperrag.retrieval.search")
            .description("PaperRAG retrieval search latency")
            .register(meterRegistry);
    }

    public void recordChatRequest() {
        chatRequests.increment();
    }

    public <T> T recordRetrievalSearch(Supplier<T> supplier) {
        return retrievalSearchTimer.record(supplier);
    }

    public void recordRateLimit(boolean allowed) {
        Counter.builder("paperrag.rate.limit.requests")
            .description("PaperRAG rate-limit decisions")
            .tag("outcome", allowed ? "allowed" : "rejected")
            .register(meterRegistry)
            .increment();
    }
}
