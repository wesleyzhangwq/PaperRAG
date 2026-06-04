package com.wesz.paperrag.eval;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;

@SpringBootTest
@AutoConfigureMockMvc
class EvalControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @Test
    void runsAndPersistsEvaluationSummary() throws Exception {
        String body = mockMvc.perform(post("/api/eval/run")
                .contentType(MediaType.APPLICATION_JSON)
                .content("""
                    {
                      "name": "smoke-eval",
                      "groundTruthChunkIds": [101, 103],
                      "retrievedChunkIds": [101, 102, 103],
                      "k": 2
                    }
                    """))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.data.runId").isNumber())
            .andExpect(jsonPath("$.data.name").value("smoke-eval"))
            .andExpect(jsonPath("$.data.recallAtK").value(0.5))
            .andExpect(jsonPath("$.data.mrr").value(1.0))
            .andExpect(jsonPath("$.data.ndcgAtK").value(0.6131))
            .andReturn()
            .getResponse()
            .getContentAsString();

        long runId = objectMapper.readTree(body).path("data").path("runId").asLong();

        mockMvc.perform(get("/api/eval/runs/{runId}", runId))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.data.runId").value(runId))
            .andExpect(jsonPath("$.data.name").value("smoke-eval"))
            .andExpect(jsonPath("$.data.recallAtK").value(0.5));
    }
}
