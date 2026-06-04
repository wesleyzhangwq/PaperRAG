package com.wesz.paperrag.ingest;

public record IngestTaskResponse(
    Long taskId,
    Long paperId,
    IngestStatus status,
    Long chunkCount,
    String errorMessage
) {

    static IngestTaskResponse from(IngestTask task, long chunkCount) {
        return new IngestTaskResponse(
            task.getId(),
            task.getPaperId(),
            task.getStatus(),
            chunkCount,
            task.getErrorMessage()
        );
    }
}
