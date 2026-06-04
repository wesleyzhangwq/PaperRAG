package com.wesz.paperrag.cache;

import java.time.Duration;
import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "app.rate-limit")
public record RateLimitProperties(Integer fixedWindowLimit, Long fixedWindowSeconds) {

    public int limit() {
        return fixedWindowLimit == null ? 2 : fixedWindowLimit;
    }

    public Duration window() {
        return Duration.ofSeconds(fixedWindowSeconds == null ? 60 : fixedWindowSeconds);
    }
}
