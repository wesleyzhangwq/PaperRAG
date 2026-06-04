package com.wesz.paperrag.vector;

import static org.hamcrest.Matchers.containsString;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.multipart;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.wesz.paperrag.ingest.IngestStatus;
import com.wesz.paperrag.ingest.IngestTask;
import com.wesz.paperrag.ingest.IngestTaskMapper;
import java.nio.charset.StandardCharsets;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.web.servlet.MockMvc;

@SpringBootTest
@AutoConfigureMockMvc
class RetrievalControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @Autowired
    private IngestTaskMapper ingestTaskMapper;

    @Test
    void searchesIndexedChunksAfterIngest() throws Exception {
        MockMultipartFile file = new MockMultipartFile(
            "file",
            "retrieval.txt",
            "text/plain",
            "alpha beta gamma delta epsilon. planning agent graph reflection.".getBytes(StandardCharsets.UTF_8)
        );

        String body = mockMvc.perform(multipart("/api/ingest/upload")
                .file(file)
                .param("title", "Retrieval Test Paper"))
            .andExpect(status().isOk())
            .andReturn()
            .getResponse()
            .getContentAsString();

        long taskId = objectMapper.readTree(body).path("data").path("taskId").asLong();
        waitForDone(taskId);

        mockMvc.perform(get("/api/retrieval/search")
                .param("query", "alpha beta")
                .param("topK", "1"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.data[0].content", containsString("alpha beta")))
            .andExpect(jsonPath("$.data[0].score").isNumber());
    }

    private void waitForDone(long taskId) throws InterruptedException {
        for (int attempt = 0; attempt < 40; attempt++) {
            IngestTask task = ingestTaskMapper.selectById(taskId);
            if (task != null && task.getStatus() == IngestStatus.DONE) {
                return;
            }
            Thread.sleep(50);
        }
        JsonNode status = objectMapper.valueToTree(ingestTaskMapper.selectById(taskId));
        throw new AssertionError("Task did not finish: " + status);
    }
}
