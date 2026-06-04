package com.wesz.paperrag.hybrid;

import java.util.Locale;

public enum HybridMode {
    HYBRID,
    KEYWORD_ONLY,
    VECTOR_ONLY;

    public static HybridMode from(String value) {
        if (value == null || value.isBlank()) {
            return HYBRID;
        }
        return switch (value.toLowerCase(Locale.ROOT)) {
            case "keyword", "keyword_only", "keyword-only" -> KEYWORD_ONLY;
            case "vector", "vector_only", "vector-only" -> VECTOR_ONLY;
            default -> HYBRID;
        };
    }
}
