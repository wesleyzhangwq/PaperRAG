package com.wesz.paperrag.chunk;

public record ChunkResponse(
    Long id,
    Long paperId,
    Integer chunkIndex,
    String content,
    Integer tokenCount
) {

    public static ChunkResponse from(PaperChunk chunk) {
        return new ChunkResponse(
            chunk.getId(),
            chunk.getPaperId(),
            chunk.getChunkIndex(),
            chunk.getContent(),
            chunk.getTokenCount()
        );
    }
}
