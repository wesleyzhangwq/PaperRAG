package com.wesz.paperrag.vector;

import com.wesz.paperrag.common.ApiResponse;
import java.util.List;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/retrieval")
public class RetrievalController {

    private final RetrievalService retrievalService;

    public RetrievalController(RetrievalService retrievalService) {
        this.retrievalService = retrievalService;
    }

    @GetMapping("/search")
    public ApiResponse<List<RetrievalResultResponse>> search(
        @RequestParam String query,
        @RequestParam(defaultValue = "5") int topK
    ) {
        return ApiResponse.ok(retrievalService.search(query, topK));
    }
}
