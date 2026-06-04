package com.wesz.paperrag.cache;

import com.wesz.paperrag.common.BusinessException;
import com.wesz.paperrag.paper.Paper;
import com.wesz.paperrag.paper.PaperMapper;
import java.time.Duration;
import java.util.Optional;
import java.util.concurrent.ThreadLocalRandom;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;

@Service
public class PaperCacheService {

    private final CacheStore cacheStore;
    private final PaperMapper paperMapper;
    private final RedisProperties redisProperties;

    public PaperCacheService(
        CacheStore cacheStore,
        PaperMapper paperMapper,
        RedisProperties redisProperties
    ) {
        this.cacheStore = cacheStore;
        this.paperMapper = paperMapper;
        this.redisProperties = redisProperties;
    }

    public CachedPaperResponse getPaper(long id) {
        String key = "paper:" + id;
        Optional<CacheLookup<CachedPaperValue>> cached = cacheStore.get(key, CachedPaperValue.class);
        if (cached.isPresent()) {
            if (cached.get().nullValue()) {
                throw notFound(id);
            }
            return CachedPaperResponse.from(cached.get().value(), true);
        }

        Paper paper = paperMapper.selectById(id);
        if (paper == null) {
            cacheStore.putNull(key, jitter(redisProperties.nullTtl()));
            throw notFound(id);
        }

        CachedPaperValue value = new CachedPaperValue(
            paper.getId(),
            paper.getTitle(),
            paper.getAbstractText(),
            paper.getAuthors()
        );
        cacheStore.put(key, value, jitter(redisProperties.defaultTtl()));
        return CachedPaperResponse.from(value, false);
    }

    private Duration jitter(Duration baseTtl) {
        double ratio = redisProperties.jitterRatio();
        if (ratio <= 0.0 || baseTtl.isZero() || baseTtl.isNegative()) {
            return baseTtl;
        }
        long seconds = Math.max(1L, baseTtl.toSeconds());
        long range = Math.max(1L, Math.round(seconds * ratio));
        long delta = ThreadLocalRandom.current().nextLong(-range, range + 1);
        return Duration.ofSeconds(Math.max(1L, seconds + delta));
    }

    private BusinessException notFound(long id) {
        return new BusinessException(HttpStatus.NOT_FOUND, "Paper " + id + " not found");
    }
}
