package com.wesz.paperrag.auth;

import static org.hamcrest.Matchers.containsStringIgnoringCase;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.wesz.paperrag.paper.Paper;
import com.wesz.paperrag.paper.PaperMapper;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;

@SpringBootTest
@AutoConfigureMockMvc
class AuthTenantControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @Autowired
    private PaperMapper paperMapper;

    @Test
    void issuesJwtAndProtectsTenantEndpoints() throws Exception {
        String token = login("alice", "password");

        mockMvc.perform(get("/api/tenant/me")
                .header("Authorization", "Bearer " + token))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.data.username").value("alice"))
            .andExpect(jsonPath("$.data.tenantId").value(1));

        mockMvc.perform(get("/api/tenant/me"))
            .andExpect(status().isUnauthorized());
    }

    @Test
    void rejectsCrossTenantPaperAccess() throws Exception {
        Paper paper = Paper.create("Tenant One Paper", "tenant only", "Alice");
        paper.setTenantId(1L);
        paperMapper.insert(paper);

        String aliceToken = login("alice", "password");
        String bobToken = login("bob", "password");

        mockMvc.perform(get("/api/tenant/papers/{id}", paper.getId())
                .header("Authorization", "Bearer " + aliceToken))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.data.id").value(paper.getId()));

        mockMvc.perform(get("/api/tenant/papers/{id}", paper.getId())
                .header("Authorization", "Bearer " + bobToken))
            .andExpect(status().isForbidden())
            .andExpect(jsonPath("$.message", containsStringIgnoringCase("tenant")));
    }

    private String login(String username, String password) throws Exception {
        String body = mockMvc.perform(post("/api/auth/login")
                .contentType(MediaType.APPLICATION_JSON)
                .content("""
                    {"username":"%s","password":"%s"}
                    """.formatted(username, password)))
            .andExpect(status().isOk())
            .andReturn()
            .getResponse()
            .getContentAsString();
        return objectMapper.readTree(body).path("data").path("token").asText();
    }
}
