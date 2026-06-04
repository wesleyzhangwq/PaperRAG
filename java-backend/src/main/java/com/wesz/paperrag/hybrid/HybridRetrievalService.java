package com.wesz.paperrag.hybrid;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.springframework.stereotype.Service;

@Service
public class HybridRetrievalService {

    private static final int RRF_K = 60;

    private final KeywordRetrievalService keywordRetrievalService;
    private final VectorRetrieverAdapter vectorRetrieverAdapter;

    public HybridRetrievalService(
        KeywordRetrievalService keywordRetrievalService,
        VectorRetrieverAdapter vectorRetrieverAdapter
    ) {
        this.keywordRetrievalService = keywordRetrievalService;
        this.vectorRetrieverAdapter = vectorRetrieverAdapter;
    }

    public HybridSearchResponse search(String query, int topK, String modeValue, boolean degradeVector) {
        HybridMode requestedMode = HybridMode.from(modeValue);
        int boundedTopK = Math.max(1, Math.min(20, topK));

        if (requestedMode == HybridMode.KEYWORD_ONLY) {
            return new HybridSearchResponse(
                HybridMode.KEYWORD_ONLY,
                false,
                direct(keywordRetrievalService.search(query, boundedTopK))
            );
        }

        if (requestedMode == HybridMode.VECTOR_ONLY && !degradeVector) {
            return new HybridSearchResponse(
                HybridMode.VECTOR_ONLY,
                false,
                direct(vectorRetrieverAdapter.search(query, boundedTopK))
            );
        }

        List<RankedChunk> keywordResults = keywordRetrievalService.search(query, boundedTopK);
        if (degradeVector) {
            return new HybridSearchResponse(
                HybridMode.KEYWORD_ONLY,
                true,
                direct(keywordResults)
            );
        }

        List<RankedChunk> vectorResults;
        boolean degraded = false;
        try {
            vectorResults = vectorRetrieverAdapter.search(query, boundedTopK);
        } catch (RuntimeException exception) {
            vectorResults = List.of();
            degraded = true;
        }

        if (requestedMode == HybridMode.VECTOR_ONLY) {
            return new HybridSearchResponse(
                degraded ? HybridMode.KEYWORD_ONLY : HybridMode.VECTOR_ONLY,
                degraded,
                direct(degraded ? keywordResults : vectorResults)
            );
        }

        List<HybridSearchResult> fused = rrf(List.of(keywordResults, vectorResults), boundedTopK);
        return new HybridSearchResponse(
            degraded ? HybridMode.KEYWORD_ONLY : HybridMode.HYBRID,
            degraded,
            degraded ? direct(keywordResults) : fused
        );
    }

    private List<HybridSearchResult> direct(List<RankedChunk> rankedChunks) {
        return rankedChunks.stream()
            .map(result -> new HybridSearchResult(
                result.chunkId(),
                result.paperId(),
                result.chunkIndex(),
                result.content(),
                result.score(),
                List.of(result.channel())
            ))
            .toList();
    }

    private List<HybridSearchResult> rrf(List<List<RankedChunk>> rankings, int topK) {
        Map<Long, Accumulator> byChunk = new LinkedHashMap<>();
        for (List<RankedChunk> ranking : rankings) {
            for (int index = 0; index < ranking.size(); index++) {
                RankedChunk item = ranking.get(index);
                Accumulator accumulator = byChunk.computeIfAbsent(
                    item.chunkId(),
                    ignored -> new Accumulator(item)
                );
                accumulator.score += 1.0 / (RRF_K + index + 1);
                accumulator.channels.add(item.channel());
            }
        }
        return byChunk.values()
            .stream()
            .sorted(Comparator
                .comparingDouble(Accumulator::score).reversed()
                .thenComparing(accumulator -> accumulator.item.chunkId()))
            .limit(topK)
            .map(Accumulator::toResult)
            .toList();
    }

    private static class Accumulator {
        private final RankedChunk item;
        private final Set<String> channels = new LinkedHashSet<>();
        private double score;

        private Accumulator(RankedChunk item) {
            this.item = item;
        }

        private double score() {
            return score;
        }

        private HybridSearchResult toResult() {
            return new HybridSearchResult(
                item.chunkId(),
                item.paperId(),
                item.chunkIndex(),
                item.content(),
                score,
                new ArrayList<>(channels)
            );
        }
    }
}
