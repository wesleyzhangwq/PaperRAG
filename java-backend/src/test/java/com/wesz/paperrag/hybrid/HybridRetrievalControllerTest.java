package com.wesz.paperrag.hybrid;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import com.wesz.paperrag.chunk.PaperChunk;
import com.wesz.paperrag.chunk.PaperChunkMapper;
import com.wesz.paperrag.paper.Paper;
import com.wesz.paperrag.paper.PaperMapper;
import com.wesz.paperrag.vector.RetrievalService;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.web.servlet.MockMvc;

@SpringBootTest
@AutoConfigureMockMvc
class HybridRetrievalControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private PaperMapper paperMapper;

    @Autowired
    private PaperChunkMapper chunkMapper;

    @Autowired
    private RetrievalService retrievalService;

    @Test
    void hybridSearchUsesRrfAndRanksLexicalMatchFirst() throws Exception {
        seedChunks();

        mockMvc.perform(get("/api/admin/eval/hybrid")
                .param("query", "rarekeyword")
                .param("topK", "2"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.data.mode").value("HYBRID"))
            .andExpect(jsonPath("$.data.degraded").value(false))
            .andExpect(jsonPath("$.data.results[0].content").value("rarekeyword rarekeyword sparse lexical matching"))
            .andExpect(jsonPath("$.data.results[0].channels[0]").exists());
    }

    @Test
    void supportsSingleRetrieverModesAndVectorDegradation() throws Exception {
        seedChunks();

        mockMvc.perform(get("/api/admin/eval/hybrid")
                .param("query", "rarekeyword")
                .param("topK", "2")
                .param("mode", "keyword"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.data.mode").value("KEYWORD_ONLY"))
            .andExpect(jsonPath("$.data.results[0].channels[0]").value("keyword"));

        mockMvc.perform(get("/api/admin/eval/hybrid")
                .param("query", "semantic")
                .param("topK", "2")
                .param("mode", "vector"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.data.mode").value("VECTOR_ONLY"))
            .andExpect(jsonPath("$.data.results[0].channels[0]").value("vector"));

        mockMvc.perform(get("/api/admin/eval/hybrid")
                .param("query", "rarekeyword")
                .param("topK", "2")
                .param("degradeVector", "true"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.data.mode").value("KEYWORD_ONLY"))
            .andExpect(jsonPath("$.data.degraded").value(true));
    }

    private void seedChunks() {
        Paper lexicalPaper = Paper.create("Lexical Paper", "keyword retrieval", "Alice");
        paperMapper.insert(lexicalPaper);
        Paper vectorPaper = Paper.create("Vector Paper", "semantic retrieval", "Bob");
        paperMapper.insert(vectorPaper);

        PaperChunk lexical = PaperChunk.create(
            lexicalPaper.getId(),
            lexicalPaper.getTenantId(),
            0,
            "rarekeyword rarekeyword sparse lexical matching"
        );
        PaperChunk semantic = PaperChunk.create(
            vectorPaper.getId(),
            vectorPaper.getTenantId(),
            0,
            "semantic neural retrieval embeddings"
        );
        chunkMapper.insert(lexical);
        chunkMapper.insert(semantic);
        retrievalService.indexChunks(List.of(lexical, semantic));
    }
}
