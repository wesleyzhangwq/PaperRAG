package com.wesz.paperrag.ingest;

import static org.assertj.core.api.Assertions.assertThat;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.multipart;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.wesz.paperrag.paper.Paper;
import com.wesz.paperrag.paper.PaperMapper;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.web.servlet.MockMvc;

@SpringBootTest
@AutoConfigureMockMvc
class IngestMessageAdminControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @Autowired
    private IngestTaskMapper ingestTaskMapper;

    @Autowired
    private PaperMapper paperMapper;

    @Test
    void uploadPublishesIngestMessageAndExposesAdminStats() throws Exception {
        MockMultipartFile file = new MockMultipartFile(
            "file",
            "message.txt",
            "text/plain",
            "message driven ingestion boundary".getBytes(StandardCharsets.UTF_8)
        );

        String body = mockMvc.perform(multipart("/api/ingest/upload")
                .file(file)
                .param("title", "Message Boundary"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.data.taskId").isNumber())
            .andReturn()
            .getResponse()
            .getContentAsString();

        long taskId = objectMapper.readTree(body).path("data").path("taskId").asLong();

        String statsBody = mockMvc.perform(get("/api/admin/ingest/messages"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.data.lastTaskId").value(taskId))
            .andReturn()
            .getResponse()
            .getContentAsString();

        JsonNode stats = objectMapper.readTree(statsBody).path("data");
        assertThat(stats.path("publishedCount").asLong()).isGreaterThanOrEqualTo(1L);
        assertThat(stats.path("consumedCount").asLong()).isGreaterThanOrEqualTo(1L);
    }

    @Test
    void invalidPdfMessageRecordsFailedTask() throws Exception {
        MockMultipartFile file = new MockMultipartFile(
            "file",
            "broken.pdf",
            "application/pdf",
            "not a valid pdf".getBytes(StandardCharsets.UTF_8)
        );

        String body = mockMvc.perform(multipart("/api/ingest/upload")
                .file(file)
                .param("title", "Broken PDF"))
            .andExpect(status().isOk())
            .andReturn()
            .getResponse()
            .getContentAsString();

        long taskId = objectMapper.readTree(body).path("data").path("taskId").asLong();

        IngestTask failed = waitForStatus(taskId, IngestStatus.FAILED);
        assertThat(failed.getErrorMessage()).contains("Unable to parse PDF upload");
    }

    @Test
    void failedTaskCanBeRepublishedThroughAdminRetry() throws Exception {
        Paper paper = Paper.create("Retry Paper", "retry payload text", "Alice");
        paperMapper.insert(paper);
        IngestTask task = IngestTask.pending(
            paper.getId(),
            paper.getTenantId(),
            "retry-" + System.nanoTime()
        );
        ingestTaskMapper.insert(task);
        task.markFailed("temporary failure", Instant.now());
        ingestTaskMapper.updateById(task);

        mockMvc.perform(post("/api/admin/ingest/tasks/{taskId}/retry", task.getId()))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.data.taskId").value(task.getId()))
            .andExpect(jsonPath("$.data.status").value("PENDING"))
            .andExpect(jsonPath("$.data.republished").value(true));

        mockMvc.perform(get("/api/admin/ingest/messages"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.data.lastTaskId").value(task.getId()));
    }

    private IngestTask waitForStatus(long taskId, IngestStatus status) throws InterruptedException {
        for (int attempt = 0; attempt < 40; attempt++) {
            IngestTask task = ingestTaskMapper.selectById(taskId);
            if (task != null && task.getStatus() == status) {
                return task;
            }
            Thread.sleep(50);
        }
        IngestTask task = ingestTaskMapper.selectById(taskId);
        throw new AssertionError("Task did not reach " + status + ". Last status: "
            + (task == null ? "missing" : task.getStatus()));
    }
}
