package com.wesz.paperrag.cache;

import java.time.Instant;

record CacheEntry(Object value, boolean nullValue, Instant expiresAt) {

    boolean expired(Instant now) {
        return !expiresAt.isAfter(now);
    }
}
