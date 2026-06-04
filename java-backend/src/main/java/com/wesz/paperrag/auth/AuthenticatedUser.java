package com.wesz.paperrag.auth;

public record AuthenticatedUser(String username, Long tenantId) {
}
