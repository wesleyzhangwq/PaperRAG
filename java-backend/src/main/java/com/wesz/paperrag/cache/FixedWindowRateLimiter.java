package com.wesz.paperrag.cache;

import java.time.Clock;
import java.time.Duration;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import org.springframework.stereotype.Component;

@Component
public class FixedWindowRateLimiter {

    private final Clock clock;
    private final ConcurrentMap<String, Window> windows = new ConcurrentHashMap<>();

    public FixedWindowRateLimiter() {
        this(Clock.systemUTC());
    }

    FixedWindowRateLimiter(Clock clock) {
        this.clock = clock;
    }

    public boolean tryAcquire(String key, int limit, Duration windowSize) {
        long now = clock.millis();
        long windowMillis = Math.max(1L, windowSize.toMillis());
        Window current = windows.compute(key, (ignored, existing) -> {
            if (existing == null || now - existing.startedAtMillis() >= windowMillis) {
                return new Window(now, 1);
            }
            return new Window(existing.startedAtMillis(), existing.count() + 1);
        });
        return current.count() <= limit;
    }

    private record Window(long startedAtMillis, int count) {
    }
}
