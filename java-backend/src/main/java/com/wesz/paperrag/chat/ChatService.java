package com.wesz.paperrag.chat;

import com.wesz.paperrag.vector.RetrievalService;
import com.wesz.paperrag.observability.ObservabilityMetrics;
import java.util.List;
import org.springframework.stereotype.Service;

@Service
public class ChatService {

    private final RetrievalService retrievalService;
    private final LlmProvider llmProvider;
    private final ObservabilityMetrics metrics;

    public ChatService(
        RetrievalService retrievalService,
        LlmProvider llmProvider,
        ObservabilityMetrics metrics
    ) {
        this.retrievalService = retrievalService;
        this.llmProvider = llmProvider;
        this.metrics = metrics;
    }

    public ChatResponse answer(String question) {
        metrics.recordChatRequest();
        List<SourceResponse> sources = retrievalService.search(question, 5)
            .stream()
            .map(SourceResponse::from)
            .toList();
        return new ChatResponse(llmProvider.answer(question, sources), sources);
    }
}
