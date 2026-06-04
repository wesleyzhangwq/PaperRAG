package com.wesz.paperrag.common;

public record ApiResponse<T>(int code, String message, T data) {

    public static <T> ApiResponse<T> ok(T data) {
        return new ApiResponse<>(0, "OK", data);
    }

    public static <T> ApiResponse<T> fail(int code, String message) {
        return new ApiResponse<>(code, message, null);
    }
}
