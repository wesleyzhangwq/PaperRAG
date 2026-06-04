package com.wesz.paperrag.vector;

import static org.assertj.core.api.Assertions.assertThat;
import static org.hamcrest.Matchers.containsString;
import static org.springframework.http.HttpMethod.POST;
import static org.springframework.http.HttpMethod.PUT;
import static org.springframework.http.MediaType.APPLICATION_JSON;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.content;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.method;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.requestTo;
import static org.springframework.test.web.client.response.MockRestResponseCreators.withSuccess;

import java.util.List;
import org.junit.jupiter.api.Test;
import org.springframework.web.client.RestTemplate;
import org.springframework.test.web.client.MockRestServiceServer;

class QdrantVectorStoreTest {

    @Test
    void upsertsPointsToQdrantCollection() {
        RestTemplate restTemplate = new RestTemplate();
        MockRestServiceServer server = MockRestServiceServer.createServer(restTemplate);
        QdrantVectorStore store = new QdrantVectorStore(
            restTemplate,
            new QdrantProperties("http://qdrant:6333", "paper_chunks", 2)
        );

        server.expect(requestTo("http://qdrant:6333/collections/paper_chunks/points?wait=true"))
            .andExpect(method(PUT))
            .andExpect(content().string(containsString("\"id\":7")))
            .andExpect(content().string(containsString("\"paperId\":11")))
            .andRespond(withSuccess("{\"status\":\"ok\"}", APPLICATION_JSON));

        store.upsert(List.of(new VectorDocument(
            7L,
            11L,
            0,
            "hybrid retrieval",
            new float[] {1.0f, 0.0f}
        )));

        server.verify();
    }

    @Test
    void mapsQdrantSearchResults() {
        RestTemplate restTemplate = new RestTemplate();
        MockRestServiceServer server = MockRestServiceServer.createServer(restTemplate);
        QdrantVectorStore store = new QdrantVectorStore(
            restTemplate,
            new QdrantProperties("http://qdrant:6333", "paper_chunks", 2)
        );

        server.expect(requestTo("http://qdrant:6333/collections/paper_chunks/points/search"))
            .andExpect(method(POST))
            .andExpect(content().string(containsString("\"limit\":1")))
            .andRespond(withSuccess("""
                {
                  "result": [
                    {
                      "id": 7,
                      "score": 0.91,
                      "payload": {
                        "paperId": 11,
                        "chunkIndex": 0,
                        "content": "hybrid retrieval"
                      }
                    }
                  ]
                }
                """, APPLICATION_JSON));

        List<VectorSearchResult> results = store.search(new float[] {1.0f, 0.0f}, 1);

        assertThat(results).hasSize(1);
        assertThat(results.getFirst().chunkId()).isEqualTo(7L);
        assertThat(results.getFirst().paperId()).isEqualTo(11L);
        assertThat(results.getFirst().score()).isEqualTo(0.91);
    }
}
