package com.wesz.paperrag.vector;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;
import org.junit.jupiter.api.Test;

class InMemoryVectorStoreTest {

    @Test
    void returnsTopKByCosineSimilarity() {
        VectorStore vectorStore = new InMemoryVectorStore();
        vectorStore.upsert(List.of(
            new VectorDocument(1L, 10L, 0, "hybrid retrieval", new float[] {1.0f, 0.0f}),
            new VectorDocument(2L, 11L, 0, "agent planning", new float[] {0.0f, 1.0f}),
            new VectorDocument(3L, 12L, 0, "retrieval evaluation", new float[] {0.8f, 0.2f})
        ));

        List<VectorSearchResult> results = vectorStore.search(new float[] {1.0f, 0.0f}, 2);

        assertThat(results).hasSize(2);
        assertThat(results.getFirst().chunkId()).isEqualTo(1L);
        assertThat(results.getFirst().score()).isGreaterThan(results.getLast().score());
        assertThat(results.getLast().chunkId()).isEqualTo(3L);
    }
}
