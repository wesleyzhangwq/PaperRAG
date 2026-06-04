package com.wesz.paperrag.observability;

import io.micrometer.prometheusmetrics.PrometheusMeterRegistry;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class PrometheusScrapeController {

    private final PrometheusMeterRegistry registry;

    public PrometheusScrapeController(PrometheusMeterRegistry registry) {
        this.registry = registry;
    }

    @GetMapping(path = "/actuator/prometheus", produces = MediaType.TEXT_PLAIN_VALUE)
    String scrape() {
        return registry.scrape();
    }
}
