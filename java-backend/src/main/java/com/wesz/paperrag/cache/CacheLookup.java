package com.wesz.paperrag.cache;

public record CacheLookup<T>(T value, boolean nullValue) {

    public static <T> CacheLookup<T> value(T value) {
        return new CacheLookup<>(value, false);
    }

    public static <T> CacheLookup<T> nullHit() {
        return new CacheLookup<>(null, true);
    }
}
