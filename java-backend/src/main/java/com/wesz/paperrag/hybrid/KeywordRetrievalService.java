package com.wesz.paperrag.hybrid;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.wesz.paperrag.chunk.PaperChunk;
import com.wesz.paperrag.chunk.PaperChunkMapper;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;
import org.springframework.stereotype.Service;

@Service
public class KeywordRetrievalService {

    private static final Pattern SPLIT_PATTERN = Pattern.compile("[^\\p{IsAlphabetic}\\p{IsDigit}]+");

    private final PaperChunkMapper chunkMapper;

    public KeywordRetrievalService(PaperChunkMapper chunkMapper) {
        this.chunkMapper = chunkMapper;
    }

    public List<RankedChunk> search(String query, int topK) {
        List<String> queryTerms = tokenize(query);
        if (queryTerms.isEmpty()) {
            return List.of();
        }
        List<PaperChunk> chunks = chunkMapper.selectList(
            new LambdaQueryWrapper<PaperChunk>().orderByAsc(PaperChunk::getId)
        );
        Map<String, Integer> documentFrequency = documentFrequency(chunks);
        int documentCount = Math.max(1, chunks.size());

        List<RankedChunk> scored = new ArrayList<>();
        for (PaperChunk chunk : chunks) {
            List<String> terms = tokenize(chunk.getContent());
            if (terms.isEmpty()) {
                continue;
            }
            Map<String, Integer> termFrequency = termFrequency(terms);
            double score = 0.0;
            for (String queryTerm : queryTerms) {
                int tf = termFrequency.getOrDefault(queryTerm, 0);
                if (tf == 0) {
                    continue;
                }
                int df = Math.max(1, documentFrequency.getOrDefault(queryTerm, 1));
                double idf = Math.log(1.0 + ((documentCount - df + 0.5) / (df + 0.5)));
                double lengthNorm = 0.75 + 0.25 * Math.min(1.0, 12.0 / terms.size());
                score += tf * idf * lengthNorm;
            }
            if (score > 0.0) {
                scored.add(new RankedChunk(
                    chunk.getId(),
                    chunk.getPaperId(),
                    chunk.getChunkIndex(),
                    chunk.getContent(),
                    score,
                    "keyword"
                ));
            }
        }
        return scored.stream()
            .sorted(Comparator
                .comparingDouble(RankedChunk::score).reversed()
                .thenComparing(RankedChunk::chunkId))
            .limit(Math.max(1, topK))
            .toList();
    }

    private Map<String, Integer> documentFrequency(List<PaperChunk> chunks) {
        Map<String, Integer> frequencies = new HashMap<>();
        for (PaperChunk chunk : chunks) {
            Set<String> uniqueTerms = new HashSet<>(tokenize(chunk.getContent()));
            for (String term : uniqueTerms) {
                frequencies.merge(term, 1, Integer::sum);
            }
        }
        return frequencies;
    }

    private Map<String, Integer> termFrequency(List<String> terms) {
        Map<String, Integer> frequencies = new HashMap<>();
        for (String term : terms) {
            frequencies.merge(term, 1, Integer::sum);
        }
        return frequencies;
    }

    private List<String> tokenize(String text) {
        if (text == null || text.isBlank()) {
            return List.of();
        }
        String[] parts = SPLIT_PATTERN.split(text.toLowerCase(Locale.ROOT));
        List<String> terms = new ArrayList<>();
        for (String part : parts) {
            if (!part.isBlank()) {
                terms.add(part);
            }
        }
        return terms;
    }
}
