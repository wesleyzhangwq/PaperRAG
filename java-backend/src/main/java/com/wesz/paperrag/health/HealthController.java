package com.wesz.paperrag.health;

import com.wesz.paperrag.common.ApiResponse;
import java.time.Instant;
import java.util.Map;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/health")
public class HealthController {

    @GetMapping
    public ApiResponse<Map<String, String>> health() {
        return ApiResponse.ok(Map.of(
            "status", "OK",
            "time", Instant.now().toString()
        ));
    }
}
