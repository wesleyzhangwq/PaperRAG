package com.wesz.paperrag.ingest;

import com.wesz.paperrag.common.ApiResponse;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/api/ingest")
public class IngestController {

    private final IngestUploadService ingestUploadService;
    private final IngestQueryService ingestQueryService;

    public IngestController(
        IngestUploadService ingestUploadService,
        IngestQueryService ingestQueryService
    ) {
        this.ingestUploadService = ingestUploadService;
        this.ingestQueryService = ingestQueryService;
    }

    @PostMapping("/upload")
    public ApiResponse<IngestUploadResponse> upload(
        @RequestParam("file") MultipartFile file,
        @RequestParam(required = false) String title
    ) {
        return ApiResponse.ok(ingestUploadService.upload(file, title));
    }

    @GetMapping("/tasks/{taskId}")
    public ApiResponse<IngestTaskResponse> getTask(@PathVariable long taskId) {
        return ApiResponse.ok(ingestQueryService.getTask(taskId));
    }
}
