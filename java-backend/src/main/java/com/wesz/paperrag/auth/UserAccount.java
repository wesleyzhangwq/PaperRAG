package com.wesz.paperrag.auth;

public record UserAccount(String username, String passwordHash, Long tenantId) {
}
