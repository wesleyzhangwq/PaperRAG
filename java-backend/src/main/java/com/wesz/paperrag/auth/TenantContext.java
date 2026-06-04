package com.wesz.paperrag.auth;

public final class TenantContext {

    private static final ThreadLocal<AuthenticatedUser> CURRENT = new ThreadLocal<>();

    private TenantContext() {
    }

    public static void set(AuthenticatedUser user) {
        CURRENT.set(user);
    }

    public static AuthenticatedUser require() {
        AuthenticatedUser user = CURRENT.get();
        if (user == null) {
            throw new IllegalStateException("No tenant context is available");
        }
        return user;
    }

    public static void clear() {
        CURRENT.remove();
    }
}
