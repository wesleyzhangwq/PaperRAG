package com.wesz.paperrag.cache;

import com.wesz.paperrag.common.ApiResponse;
import com.wesz.paperrag.common.BusinessException;
import com.wesz.paperrag.observability.ObservabilityMetrics;
import java.util.Map;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/cache")
public class CacheController {

    private final PaperCacheService paperCacheService;
    private final FixedWindowRateLimiter rateLimiter;
    private final RateLimitProperties rateLimitProperties;
    private final ObservabilityMetrics metrics;

    public CacheController(
        PaperCacheService paperCacheService,
        FixedWindowRateLimiter rateLimiter,
        RateLimitProperties rateLimitProperties,
        ObservabilityMetrics metrics
    ) {
        this.paperCacheService = paperCacheService;
        this.rateLimiter = rateLimiter;
        this.rateLimitProperties = rateLimitProperties;
        this.metrics = metrics;
    }

    @GetMapping("/papers/{id}")
    ApiResponse<CachedPaperResponse> getPaper(@PathVariable long id) {
        return ApiResponse.ok(paperCacheService.getPaper(id));
    }

    @GetMapping("/rate-limited-ping")
    ApiResponse<Map<String, Boolean>> rateLimitedPing(
        @RequestHeader(name = "X-Rate-Key", defaultValue = "anonymous") String key
    ) {
        boolean allowed = rateLimiter.tryAcquire(
            key,
            rateLimitProperties.limit(),
            rateLimitProperties.window()
        );
        metrics.recordRateLimit(allowed);
        if (!allowed) {
            throw new BusinessException(HttpStatus.TOO_MANY_REQUESTS, "rate limit exceeded");
        }
        return ApiResponse.ok(Map.of("ok", true));
    }
}
