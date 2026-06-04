package com.wesz.paperrag.auth;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.Base64;
import java.util.Map;
import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

@Service
public class JwtService {

    private final ObjectMapper objectMapper;
    private final byte[] secret;

    public JwtService(
        ObjectMapper objectMapper,
        @Value("${app.auth.jwt-secret:paperrag-dev-secret}") String secret
    ) {
        this.objectMapper = objectMapper;
        this.secret = secret.getBytes(StandardCharsets.UTF_8);
    }

    public String issue(UserAccount account) {
        try {
            String header = encode(objectMapper.writeValueAsBytes(Map.of("alg", "HS256", "typ", "JWT")));
            String payload = encode(objectMapper.writeValueAsBytes(Map.of(
                "sub", account.username(),
                "tenantId", account.tenantId(),
                "exp", Instant.now().plusSeconds(3600).getEpochSecond()
            )));
            return header + "." + payload + "." + sign(header + "." + payload);
        } catch (Exception exception) {
            throw new IllegalStateException("Unable to issue JWT", exception);
        }
    }

    public AuthenticatedUser parse(String token) {
        try {
            String[] parts = token.split("\\.");
            if (parts.length != 3 || !sign(parts[0] + "." + parts[1]).equals(parts[2])) {
                return null;
            }
            JsonNode payload = objectMapper.readTree(Base64.getUrlDecoder().decode(parts[1]));
            if (payload.path("exp").asLong() < Instant.now().getEpochSecond()) {
                return null;
            }
            return new AuthenticatedUser(payload.path("sub").asText(), payload.path("tenantId").asLong());
        } catch (Exception exception) {
            return null;
        }
    }

    private String encode(byte[] bytes) {
        return Base64.getUrlEncoder().withoutPadding().encodeToString(bytes);
    }

    private String sign(String value) throws Exception {
        Mac mac = Mac.getInstance("HmacSHA256");
        mac.init(new SecretKeySpec(secret, "HmacSHA256"));
        return encode(mac.doFinal(value.getBytes(StandardCharsets.UTF_8)));
    }
}
