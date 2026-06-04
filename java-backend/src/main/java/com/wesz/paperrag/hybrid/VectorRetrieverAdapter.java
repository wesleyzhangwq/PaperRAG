package com.wesz.paperrag.hybrid;

import com.wesz.paperrag.vector.RetrievalResultResponse;
import com.wesz.paperrag.vector.RetrievalService;
import java.util.List;
import org.springframework.stereotype.Service;

@Service
public class VectorRetrieverAdapter {

    private final RetrievalService retrievalService;

    public VectorRetrieverAdapter(RetrievalService retrievalService) {
        this.retrievalService = retrievalService;
    }

    public List<RankedChunk> search(String query, int topK) {
        return retrievalService.search(query, topK)
            .stream()
            .map(this::from)
            .toList();
    }

    private RankedChunk from(RetrievalResultResponse result) {
        return new RankedChunk(
            result.chunkId(),
            result.paperId(),
            result.chunkIndex(),
            result.content(),
            result.score(),
            "vector"
        );
    }
}
