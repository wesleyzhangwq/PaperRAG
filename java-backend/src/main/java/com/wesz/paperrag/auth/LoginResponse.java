package com.wesz.paperrag.auth;

public record LoginResponse(String token, String username, Long tenantId) {
}
