package com.wesz.paperrag.ingest;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

class DocumentChunkerTest {

    @Test
    void chunksTextIntoStableWindows() {
        DocumentChunker chunker = new DocumentChunker(5);

        var chunks = chunker.chunk("one two three four five six seven eight nine");

        assertThat(chunks).hasSize(2);
        assertThat(chunks.getFirst().chunkIndex()).isZero();
        assertThat(chunks.getFirst().content()).isEqualTo("one two three four five");
        assertThat(chunks.getLast().chunkIndex()).isEqualTo(1);
        assertThat(chunks.getLast().content()).isEqualTo("six seven eight nine");
    }
}
