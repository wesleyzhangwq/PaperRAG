package com.wesz.paperrag.ingest;

import java.util.ArrayList;
import java.util.List;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

@Component
public class DocumentChunker {

    private final int maxTokens;

    public DocumentChunker(@Value("${app.ingest.chunk-size-tokens:220}") int maxTokens) {
        this.maxTokens = maxTokens;
    }

    public List<DocumentChunk> chunk(String text) {
        if (text == null || text.isBlank()) {
            return List.of();
        }
        String[] words = text.trim().split("\\s+");
        List<DocumentChunk> chunks = new ArrayList<>();
        for (int start = 0; start < words.length; start += maxTokens) {
            int end = Math.min(start + maxTokens, words.length);
            chunks.add(new DocumentChunk(chunks.size(), String.join(" ", slice(words, start, end))));
        }
        return chunks;
    }

    private List<String> slice(String[] words, int start, int end) {
        List<String> values = new ArrayList<>();
        for (int index = start; index < end; index++) {
            values.add(words[index]);
        }
        return values;
    }
}
