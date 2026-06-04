package com.wesz.paperrag.ingest;

import com.baomidou.mybatisplus.annotation.EnumValue;
import com.fasterxml.jackson.annotation.JsonValue;

public enum IngestStatus {
    PENDING("PENDING"),
    PARSING("PARSING"),
    EMBEDDING("EMBEDDING"),
    DONE("DONE"),
    FAILED("FAILED");

    @EnumValue
    private final String value;

    IngestStatus(String value) {
        this.value = value;
    }

    @JsonValue
    public String value() {
        return value;
    }
}
