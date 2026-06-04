package com.wesz.paperrag.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "app.paper")
public record PaperProperties(int maxUploadMb) {
}
