package com.wesz.paperrag.hybrid;

import com.wesz.paperrag.common.ApiResponse;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/admin/eval")
public class HybridRetrievalController {

    private final HybridRetrievalService hybridRetrievalService;

    public HybridRetrievalController(HybridRetrievalService hybridRetrievalService) {
        this.hybridRetrievalService = hybridRetrievalService;
    }

    @GetMapping("/hybrid")
    public ApiResponse<HybridSearchResponse> hybrid(
        @RequestParam String query,
        @RequestParam(defaultValue = "5") int topK,
        @RequestParam(defaultValue = "hybrid") String mode,
        @RequestParam(defaultValue = "false") boolean degradeVector
    ) {
        return ApiResponse.ok(hybridRetrievalService.search(query, topK, mode, degradeVector));
    }
}
