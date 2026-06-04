package com.wesz.paperrag.cache;

import java.time.Duration;
import java.time.Instant;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import org.springframework.stereotype.Component;

@Component
public class InMemoryCacheStore implements CacheStore {

    private final ConcurrentMap<String, CacheEntry> entries = new ConcurrentHashMap<>();

    @Override
    public <T> Optional<CacheLookup<T>> get(String key, Class<T> type) {
        CacheEntry entry = entries.get(key);
        if (entry == null) {
            return Optional.empty();
        }
        if (entry.expired(Instant.now())) {
            entries.remove(key, entry);
            return Optional.empty();
        }
        if (entry.nullValue()) {
            return Optional.of(CacheLookup.nullHit());
        }
        Object value = entry.value();
        if (!type.isInstance(value)) {
            return Optional.empty();
        }
        return Optional.of(CacheLookup.value(type.cast(value)));
    }

    @Override
    public void put(String key, Object value, Duration ttl) {
        entries.put(key, new CacheEntry(value, false, Instant.now().plus(ttl)));
    }

    @Override
    public void putNull(String key, Duration ttl) {
        entries.put(key, new CacheEntry(null, true, Instant.now().plus(ttl)));
    }
}
