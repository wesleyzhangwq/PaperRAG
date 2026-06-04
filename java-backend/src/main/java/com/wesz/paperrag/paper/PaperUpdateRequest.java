package com.wesz.paperrag.paper;

import jakarta.validation.constraints.NotBlank;

public record PaperUpdateRequest(
    @NotBlank(message = "title must not be blank")
    String title,
    String abstractText,
    String authors
) {
}
