package com.wesz.paperrag.cache;

import java.time.Duration;
import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "app.redis")
public record RedisProperties(
    String url,
    Long defaultTtlSeconds,
    Long nullTtlSeconds,
    Double ttlJitterRatio
) {

    public Duration defaultTtl() {
        return Duration.ofSeconds(defaultTtlSeconds == null ? 300 : defaultTtlSeconds);
    }

    public Duration nullTtl() {
        return Duration.ofSeconds(nullTtlSeconds == null ? 30 : nullTtlSeconds);
    }

    public double jitterRatio() {
        if (ttlJitterRatio == null) {
            return 0.1;
        }
        return Math.max(0.0, Math.min(0.5, ttlJitterRatio));
    }
}
