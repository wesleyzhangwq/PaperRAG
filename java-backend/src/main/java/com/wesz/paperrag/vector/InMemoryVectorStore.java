package com.wesz.paperrag.vector;

import java.util.Comparator;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;

@Component
@ConditionalOnProperty(name = "app.vector.store", havingValue = "memory", matchIfMissing = true)
public class InMemoryVectorStore implements VectorStore {

    private final ConcurrentMap<Long, VectorDocument> documents = new ConcurrentHashMap<>();

    @Override
    public void upsert(List<VectorDocument> values) {
        for (VectorDocument document : values) {
            documents.put(document.chunkId(), document);
        }
    }

    @Override
    public List<VectorSearchResult> search(float[] queryVector, int topK) {
        return documents.values()
            .stream()
            .map(document -> new VectorSearchResult(
                document.chunkId(),
                document.paperId(),
                document.chunkIndex(),
                document.content(),
                cosine(queryVector, document.vector())
            ))
            .sorted(Comparator.comparing(VectorSearchResult::score).reversed())
            .limit(topK)
            .toList();
    }

    private double cosine(float[] left, float[] right) {
        double dot = 0;
        double leftNorm = 0;
        double rightNorm = 0;
        int limit = Math.min(left.length, right.length);
        for (int index = 0; index < limit; index++) {
            dot += left[index] * right[index];
            leftNorm += left[index] * left[index];
            rightNorm += right[index] * right[index];
        }
        if (leftNorm == 0 || rightNorm == 0) {
            return 0;
        }
        return dot / (Math.sqrt(leftNorm) * Math.sqrt(rightNorm));
    }
}
