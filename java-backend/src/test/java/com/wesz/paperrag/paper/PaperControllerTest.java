package com.wesz.paperrag.paper;

import static org.hamcrest.Matchers.containsStringIgnoringCase;
import static org.hamcrest.Matchers.greaterThanOrEqualTo;
import static org.hamcrest.Matchers.not;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.delete;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.put;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.header;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.time.Instant;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;

@SpringBootTest
@AutoConfigureMockMvc
class PaperControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @Test
    void createsReadsUpdatesAndDeletesPaper() throws Exception {
        long paperId = createPaper("Agentic RAG for Scientific Papers", "adaptive retrieval", "Alice, Bob");

        mockMvc.perform(get("/api/papers/{id}", paperId))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.code").value(0))
            .andExpect(jsonPath("$.data.id").value(paperId))
            .andExpect(jsonPath("$.data.title").value("Agentic RAG for Scientific Papers"))
            .andExpect(jsonPath("$.data.abstractText").value("adaptive retrieval"))
            .andExpect(jsonPath("$.data.authors").value("Alice, Bob"));

        mockMvc.perform(put("/api/papers/{id}", paperId)
                .contentType(MediaType.APPLICATION_JSON)
                .content("""
                    {
                      "title": "Updated Agentic RAG",
                      "abstractText": "reflection and retrieval",
                      "authors": "Alice"
                    }
                    """))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.data.title").value("Updated Agentic RAG"))
            .andExpect(jsonPath("$.data.abstractText").value("reflection and retrieval"))
            .andExpect(jsonPath("$.data.authors").value("Alice"));

        mockMvc.perform(delete("/api/papers/{id}", paperId))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.code").value(0))
            .andExpect(jsonPath("$.data.deleted").value(true));

        mockMvc.perform(get("/api/papers/{id}", paperId))
            .andExpect(status().isNotFound())
            .andExpect(jsonPath("$.code").value(404))
            .andExpect(jsonPath("$.message", containsStringIgnoringCase("not found")));
    }

    @Test
    void listsPapersByTitleNewestFirst() throws Exception {
        long olderId = createPaper("Hybrid Retrieval for PaperRAG", "bm25 vector", "Carol");
        Thread.sleep(5);
        long newerId = createPaper("PaperRAG Hybrid Evaluation", "metrics", "Dave");
        createPaper("Unrelated LLM Systems", "planning", "Eve");

        mockMvc.perform(get("/api/papers").param("title", "hybrid"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.data.length()", greaterThanOrEqualTo(2)))
            .andExpect(jsonPath("$.data[0].id").value(newerId))
            .andExpect(jsonPath("$.data[0].title").value("PaperRAG Hybrid Evaluation"))
            .andExpect(jsonPath("$.data[1].id").value(olderId))
            .andExpect(jsonPath("$.data[1].title").value("Hybrid Retrieval for PaperRAG"))
            .andExpect(jsonPath("$.data[*].title", not(containsStringIgnoringCase("Unrelated"))));
    }

    @Test
    void paginatesPapersByTitleAndStatus() throws Exception {
        String prefix = "Paged Hybrid " + System.nanoTime();
        createPaper(prefix + " Old", "first", "Alice");
        Thread.sleep(5);
        createPaper(prefix + " Middle", "second", "Bob");
        Thread.sleep(5);
        long newestId = createPaper(prefix + " New", "third", "Carol");

        mockMvc.perform(get("/api/papers/page")
                .param("title", prefix)
                .param("status", "PENDING")
                .param("page", "1")
                .param("size", "2"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.data.page").value(1))
            .andExpect(jsonPath("$.data.size").value(2))
            .andExpect(jsonPath("$.data.total").value(3))
            .andExpect(jsonPath("$.data.items.length()").value(2))
            .andExpect(jsonPath("$.data.items[0].id").value(newestId))
            .andExpect(jsonPath("$.data.items[0].status").value("PENDING"));
    }

    @Test
    void filtersPagedPapersByCreatedAtRange() throws Exception {
        createPaper("Created Range Old", "first", "Alice");
        Thread.sleep(50);
        JsonNode target = createPaperNode("Created Range Target", "second", "Bob");
        Thread.sleep(50);
        createPaper("Created Range New", "third", "Carol");

        Instant targetCreatedAt = Instant.parse(target.path("createdAt").asText());

        mockMvc.perform(get("/api/papers/page")
                .param("title", "Created Range")
                .param("createdFrom", targetCreatedAt.minusMillis(10).toString())
                .param("createdTo", targetCreatedAt.plusMillis(10).toString())
                .param("page", "1")
                .param("size", "10"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.data.total").value(1))
            .andExpect(jsonPath("$.data.items[0].id").value(target.path("id").asLong()))
            .andExpect(jsonPath("$.data.items[0].title").value("Created Range Target"));
    }

    @Test
    void rejectsBlankTitle() throws Exception {
        mockMvc.perform(post("/api/papers")
                .contentType(MediaType.APPLICATION_JSON)
                .content("""
                    {
                      "title": " ",
                      "abstractText": "missing title",
                      "authors": "Alice"
                    }
                    """))
            .andExpect(status().isBadRequest())
            .andExpect(jsonPath("$.code").value(400))
            .andExpect(jsonPath("$.message", containsStringIgnoringCase("title")));
    }

    @Test
    void returnsNotFoundEnvelopeForMissingPaper() throws Exception {
        mockMvc.perform(get("/api/papers/{id}", 999999L))
            .andExpect(status().isNotFound())
            .andExpect(jsonPath("$.code").value(404))
            .andExpect(jsonPath("$.message", containsStringIgnoringCase("not found")));
    }

    @Test
    void echoesTraceIdHeader() throws Exception {
        mockMvc.perform(get("/api/papers").header("X-Trace-Id", "trace-test-123"))
            .andExpect(header().string("X-Trace-Id", "trace-test-123"));
    }

    private long createPaper(String title, String abstractText, String authors) throws Exception {
        return createPaperNode(title, abstractText, authors).path("id").asLong();
    }

    private JsonNode createPaperNode(String title, String abstractText, String authors) throws Exception {
        String response = mockMvc.perform(post("/api/papers")
                .contentType(MediaType.APPLICATION_JSON)
                .content("""
                    {
                      "title": "%s",
                      "abstractText": "%s",
                      "authors": "%s"
                    }
                    """.formatted(title, abstractText, authors)))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.code").value(0))
            .andReturn()
            .getResponse()
            .getContentAsString();

        JsonNode root = objectMapper.readTree(response);
        return root.path("data");
    }
}
