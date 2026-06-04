package com.wesz.paperrag.common;

import java.util.Comparator;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.FieldError;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

@RestControllerAdvice
public class GlobalExceptionHandler {

    @ExceptionHandler(BusinessException.class)
    ResponseEntity<ApiResponse<Void>> handleBusiness(BusinessException exception) {
        HttpStatus status = exception.status();
        return ResponseEntity
            .status(status)
            .body(ApiResponse.fail(status.value(), exception.getMessage()));
    }

    @ExceptionHandler(MethodArgumentNotValidException.class)
    ResponseEntity<ApiResponse<Void>> handleValidation(MethodArgumentNotValidException exception) {
        String message = exception.getBindingResult()
            .getFieldErrors()
            .stream()
            .min(Comparator.comparing(FieldError::getField))
            .map(error -> error.getField() + ": " + error.getDefaultMessage())
            .orElse("validation failed");
        return ResponseEntity
            .badRequest()
            .body(ApiResponse.fail(HttpStatus.BAD_REQUEST.value(), message));
    }

    @ExceptionHandler(Exception.class)
    ResponseEntity<ApiResponse<Void>> handleUnexpected(Exception exception) {
        return ResponseEntity
            .status(HttpStatus.INTERNAL_SERVER_ERROR)
            .body(ApiResponse.fail(
                HttpStatus.INTERNAL_SERVER_ERROR.value(),
                "internal server error"
            ));
    }
}
