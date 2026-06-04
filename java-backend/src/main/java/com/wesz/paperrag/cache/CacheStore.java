package com.wesz.paperrag.cache;

import java.time.Duration;
import java.util.Optional;

public interface CacheStore {

    <T> Optional<CacheLookup<T>> get(String key, Class<T> type);

    void put(String key, Object value, Duration ttl);

    void putNull(String key, Duration ttl);
}
