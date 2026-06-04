package com.wesz.paperrag.paper;

import java.time.Instant;

public record PaperResponse(
    Long id,
    String title,
    String abstractText,
    String authors,
    PaperStatus status,
    Instant createdAt,
    Instant updatedAt
) {

    static PaperResponse from(Paper paper) {
        return new PaperResponse(
            paper.getId(),
            paper.getTitle(),
            paper.getAbstractText(),
            paper.getAuthors(),
            paper.getStatus(),
            paper.getCreatedAt(),
            paper.getUpdatedAt()
        );
    }
}
