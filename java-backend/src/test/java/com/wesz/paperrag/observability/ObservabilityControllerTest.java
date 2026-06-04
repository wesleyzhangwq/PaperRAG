package com.wesz.paperrag.observability;

import static org.hamcrest.Matchers.containsString;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.content;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;

@SpringBootTest
@AutoConfigureMockMvc
class ObservabilityControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @Test
    void actuatorHealthIsAvailable() throws Exception {
        mockMvc.perform(get("/actuator/health"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.status").value("UP"));
    }

    @Test
    void exposesPaperRagMetricsForChatRetrievalAndRateLimit() throws Exception {
        mockMvc.perform(post("/api/chat")
                .contentType(MediaType.APPLICATION_JSON)
                .content("""
                    {"question":"metrics check"}
                    """))
            .andExpect(status().isOk());
        mockMvc.perform(get("/api/retrieval/search")
                .param("query", "metrics")
                .param("topK", "1"))
            .andExpect(status().isOk());
        mockMvc.perform(get("/api/cache/rate-limited-ping")
                .header("X-Rate-Key", "metrics-" + System.nanoTime()))
            .andExpect(status().isOk());

        mockMvc.perform(get("/actuator/prometheus"))
            .andExpect(status().isOk())
            .andExpect(content().string(containsString("paperrag_chat_requests_total")))
            .andExpect(content().string(containsString("paperrag_retrieval_search_seconds_count")))
            .andExpect(content().string(containsString("paperrag_rate_limit_requests_total")));
    }
}
