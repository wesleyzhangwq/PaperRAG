package com.wesz.paperrag.ingest;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "app.rocketmq")
public record RocketMqProperties(String nameServer, String ingestTopic, String consumerGroup) {
}
