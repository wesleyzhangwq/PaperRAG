package com.wesz.paperrag.vector;

import com.wesz.paperrag.chunk.PaperChunk;
import com.wesz.paperrag.observability.ObservabilityMetrics;
import java.util.List;
import org.springframework.stereotype.Service;

@Service
public class RetrievalService {

    private final EmbeddingModel embeddingModel;
    private final VectorStore vectorStore;
    private final ObservabilityMetrics metrics;

    public RetrievalService(
        EmbeddingModel embeddingModel,
        VectorStore vectorStore,
        ObservabilityMetrics metrics
    ) {
        this.embeddingModel = embeddingModel;
        this.vectorStore = vectorStore;
        this.metrics = metrics;
    }

    public void indexChunks(List<PaperChunk> chunks) {
        vectorStore.upsert(chunks.stream()
            .map(chunk -> new VectorDocument(
                chunk.getId(),
                chunk.getPaperId(),
                chunk.getChunkIndex(),
                chunk.getContent(),
                embeddingModel.embed(chunk.getContent())
            ))
            .toList());
    }

    public List<RetrievalResultResponse> search(String query, int topK) {
        return metrics.recordRetrievalSearch(() -> vectorStore.search(embeddingModel.embed(query), topK)
            .stream()
            .map(RetrievalResultResponse::from)
            .toList());
    }
}
