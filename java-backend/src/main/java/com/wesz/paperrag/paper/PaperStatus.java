package com.wesz.paperrag.paper;

import com.baomidou.mybatisplus.annotation.EnumValue;
import com.fasterxml.jackson.annotation.JsonValue;

public enum PaperStatus {
    PENDING("PENDING"),
    PARSING("PARSING"),
    EMBEDDING("EMBEDDING"),
    DONE("DONE"),
    FAILED("FAILED");

    @EnumValue
    private final String value;

    PaperStatus(String value) {
        this.value = value;
    }

    @JsonValue
    public String value() {
        return value;
    }
}
