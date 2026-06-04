package com.wesz.paperrag.chat;

import jakarta.validation.constraints.NotBlank;

public record ChatRequest(@NotBlank String question) {
}
