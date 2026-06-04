package com.wesz.paperrag.chat;

import static org.hamcrest.Matchers.containsString;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.multipart;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.content;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.wesz.paperrag.ingest.IngestStatus;
import com.wesz.paperrag.ingest.IngestTask;
import com.wesz.paperrag.ingest.IngestTaskMapper;
import java.nio.charset.StandardCharsets;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.web.servlet.MockMvc;

@SpringBootTest
@AutoConfigureMockMvc
class ChatControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @Autowired
    private IngestTaskMapper ingestTaskMapper;

    @Test
    void answersQuestionWithRetrievedSources() throws Exception {
        uploadAndWait("rag-chat.txt", "hybrid retrieval combines vector recall with keyword ranking");

        mockMvc.perform(post("/api/chat")
                .contentType(MediaType.APPLICATION_JSON)
                .content("""
                    {"question":"How does hybrid retrieval work?"}
                    """))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.data.answer", containsString("[chunk:")))
            .andExpect(jsonPath("$.data.sources[0].content", containsString("hybrid retrieval")));
    }

    @Test
    void streamsSourceTokenAndDoneEvents() throws Exception {
        uploadAndWait("rag-stream.txt", "streaming answers include source cards and token events");

        mockMvc.perform(post("/api/chat/stream")
                .contentType(MediaType.APPLICATION_JSON)
                .content("""
                    {"question":"What does streaming include?"}
                    """))
            .andExpect(status().isOk())
            .andExpect(content().contentTypeCompatibleWith(MediaType.TEXT_EVENT_STREAM))
            .andExpect(content().string(containsString("event: sources")))
            .andExpect(content().string(containsString("event: token")))
            .andExpect(content().string(containsString("event: done")));
    }

    private void uploadAndWait(String filename, String content) throws Exception {
        MockMultipartFile file = new MockMultipartFile(
            "file",
            filename,
            "text/plain",
            content.getBytes(StandardCharsets.UTF_8)
        );
        String body = mockMvc.perform(multipart("/api/ingest/upload")
                .file(file)
                .param("title", filename))
            .andExpect(status().isOk())
            .andReturn()
            .getResponse()
            .getContentAsString();
        long taskId = objectMapper.readTree(body).path("data").path("taskId").asLong();
        for (int attempt = 0; attempt < 40; attempt++) {
            IngestTask task = ingestTaskMapper.selectById(taskId);
            if (task != null && task.getStatus() == IngestStatus.DONE) {
                return;
            }
            Thread.sleep(50);
        }
        throw new AssertionError("Upload task did not finish: " + taskId);
    }
}
