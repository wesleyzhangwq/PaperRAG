package com.wesz.paperrag.ingest;

import static org.assertj.core.api.Assertions.assertThat;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.multipart;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.wesz.paperrag.chunk.PaperChunk;
import com.wesz.paperrag.chunk.PaperChunkMapper;
import java.nio.charset.StandardCharsets;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.web.servlet.MockMvc;

@SpringBootTest
@AutoConfigureMockMvc
class IngestControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @Autowired
    private IngestTaskMapper ingestTaskMapper;

    @Autowired
    private PaperChunkMapper paperChunkMapper;

    @Test
    void uploadsDocumentAndPersistsChunksAsynchronously() throws Exception {
        MockMultipartFile file = new MockMultipartFile(
            "file",
            "rag.txt",
            "text/plain",
            "alpha beta gamma delta epsilon zeta eta theta iota kappa".getBytes(StandardCharsets.UTF_8)
        );

        String body = mockMvc.perform(multipart("/api/ingest/upload")
                .file(file)
                .param("title", "Uploaded RAG Paper"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.data.paperId").isNumber())
            .andExpect(jsonPath("$.data.taskId").isNumber())
            .andExpect(jsonPath("$.data.status").value("PENDING"))
            .andReturn()
            .getResponse()
            .getContentAsString();

        JsonNode data = objectMapper.readTree(body).path("data");
        long paperId = data.path("paperId").asLong();
        long taskId = data.path("taskId").asLong();

        waitForDone(taskId);

        Long chunkCount = paperChunkMapper.selectCount(
            new LambdaQueryWrapper<PaperChunk>().eq(PaperChunk::getPaperId, paperId)
        );
        assertThat(chunkCount).isEqualTo(2);

        mockMvc.perform(get("/api/ingest/tasks/{taskId}", taskId))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.data.taskId").value(taskId))
            .andExpect(jsonPath("$.data.paperId").value(paperId))
            .andExpect(jsonPath("$.data.status").value("DONE"))
            .andExpect(jsonPath("$.data.chunkCount").value(2));
    }

    private void waitForDone(long taskId) throws InterruptedException {
        for (int attempt = 0; attempt < 40; attempt++) {
            IngestTask task = ingestTaskMapper.selectById(taskId);
            if (task != null && task.getStatus() == IngestStatus.DONE) {
                return;
            }
            Thread.sleep(50);
        }
        IngestTask task = ingestTaskMapper.selectById(taskId);
        throw new AssertionError("Task did not finish. Last status: "
            + (task == null ? "missing" : task.getStatus()));
    }
}
