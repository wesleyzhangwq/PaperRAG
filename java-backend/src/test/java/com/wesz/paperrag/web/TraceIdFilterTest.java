package com.wesz.paperrag.web;

import static org.hamcrest.Matchers.matchesPattern;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.header;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.web.servlet.MockMvc;

@SpringBootTest
@AutoConfigureMockMvc
class TraceIdFilterTest {

    @Autowired
    private MockMvc mockMvc;

    @Test
    void createsTraceIdHeaderWhenRequestDoesNotProvideOne() throws Exception {
        mockMvc.perform(get("/api/health"))
            .andExpect(status().isOk())
            .andExpect(header().string(
                TraceIdFilter.TRACE_HEADER,
                matchesPattern("[0-9a-fA-F-]{36}")
            ));
    }
}
