package com.wesz.paperrag.cache;

public record CachedPaperResponse(
    Long id,
    String title,
    String abstractText,
    String authors,
    boolean cacheHit
) {

    static CachedPaperResponse from(CachedPaperValue value, boolean cacheHit) {
        return new CachedPaperResponse(
            value.id(),
            value.title(),
            value.abstractText(),
            value.authors(),
            cacheHit
        );
    }
}
