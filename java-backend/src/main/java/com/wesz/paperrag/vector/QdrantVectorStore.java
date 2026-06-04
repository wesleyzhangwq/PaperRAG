package com.wesz.paperrag.vector;

import com.fasterxml.jackson.databind.JsonNode;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate;

@Component
@ConditionalOnProperty(name = "app.vector.store", havingValue = "qdrant")
public class QdrantVectorStore implements VectorStore {

    private final RestTemplate restTemplate;
    private final QdrantProperties properties;

    public QdrantVectorStore(RestTemplate restTemplate, QdrantProperties properties) {
        this.restTemplate = restTemplate;
        this.properties = properties;
    }

    @Override
    public void upsert(List<VectorDocument> documents) {
        List<Map<String, Object>> points = documents.stream()
            .map(document -> Map.<String, Object>of(
                "id", document.chunkId(),
                "vector", toList(document.vector()),
                "payload", Map.of(
                    "paperId", document.paperId(),
                    "chunkIndex", document.chunkIndex(),
                    "content", document.content()
                )
            ))
            .toList();
        restTemplate.put(url("/collections/{collection}/points?wait=true"), Map.of("points", points));
    }

    @Override
    public List<VectorSearchResult> search(float[] queryVector, int topK) {
        ResponseEntity<JsonNode> response = restTemplate.postForEntity(
            url("/collections/{collection}/points/search"),
            Map.of(
                "vector", toList(queryVector),
                "limit", topK,
                "with_payload", true
            ),
            JsonNode.class
        );
        JsonNode result = response.getBody() == null ? null : response.getBody().path("result");
        if (result == null || !result.isArray()) {
            return List.of();
        }
        List<VectorSearchResult> values = new ArrayList<>();
        for (JsonNode node : result) {
            JsonNode payload = node.path("payload");
            values.add(new VectorSearchResult(
                node.path("id").asLong(),
                payload.path("paperId").asLong(),
                payload.path("chunkIndex").asInt(),
                payload.path("content").asText(),
                node.path("score").asDouble()
            ));
        }
        return values;
    }

    private String url(String path) {
        return properties.url() + path.replace("{collection}", properties.collection());
    }

    private List<Float> toList(float[] vector) {
        List<Float> values = new ArrayList<>(vector.length);
        for (float value : vector) {
            values.add(value);
        }
        return values;
    }
}
