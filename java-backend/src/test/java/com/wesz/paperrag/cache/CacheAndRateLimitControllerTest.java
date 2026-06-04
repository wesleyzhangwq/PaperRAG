package com.wesz.paperrag.cache;

import static org.hamcrest.Matchers.containsStringIgnoringCase;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import com.wesz.paperrag.paper.Paper;
import com.wesz.paperrag.paper.PaperMapper;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.web.servlet.MockMvc;

@SpringBootTest
@AutoConfigureMockMvc
class CacheAndRateLimitControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private PaperMapper paperMapper;

    @Test
    void cachesPaperDetailById() throws Exception {
        Paper paper = Paper.create("Cached Paper", "cache aside", "Alice");
        paperMapper.insert(paper);

        mockMvc.perform(get("/api/cache/papers/{id}", paper.getId()))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.data.title").value("Cached Paper"))
            .andExpect(jsonPath("$.data.cacheHit").value(false));

        paper.setTitle("Changed In Database");
        paperMapper.updateById(paper);

        mockMvc.perform(get("/api/cache/papers/{id}", paper.getId()))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.data.title").value("Cached Paper"))
            .andExpect(jsonPath("$.data.cacheHit").value(true));
    }

    @Test
    void rejectsRequestsAfterFixedWindowLimit() throws Exception {
        String key = "rl-" + System.nanoTime();

        mockMvc.perform(get("/api/cache/rate-limited-ping").header("X-Rate-Key", key))
            .andExpect(status().isOk());
        mockMvc.perform(get("/api/cache/rate-limited-ping").header("X-Rate-Key", key))
            .andExpect(status().isOk());
        mockMvc.perform(get("/api/cache/rate-limited-ping").header("X-Rate-Key", key))
            .andExpect(status().isTooManyRequests())
            .andExpect(jsonPath("$.message", containsStringIgnoringCase("rate limit")));
    }
}
