package com.wesz.paperrag.ingest;

import com.wesz.paperrag.common.ApiResponse;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/admin/ingest")
public class IngestAdminController {

    private final IngestMessagePublisher publisher;
    private final IngestRetryService retryService;

    public IngestAdminController(IngestMessagePublisher publisher, IngestRetryService retryService) {
        this.publisher = publisher;
        this.retryService = retryService;
    }

    @GetMapping("/messages")
    public ApiResponse<IngestMessageStats> messages() {
        return ApiResponse.ok(publisher.stats());
    }

    @PostMapping("/tasks/{taskId}/retry")
    public ApiResponse<IngestRetryResponse> retry(@PathVariable long taskId) {
        return ApiResponse.ok(retryService.retry(taskId));
    }
}
