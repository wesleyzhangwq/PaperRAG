package com.wesz.paperrag.paper;

import com.wesz.paperrag.common.ApiResponse;
import com.wesz.paperrag.common.PageResponse;
import jakarta.validation.Valid;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/papers")
public class PaperController {

    private final PaperService paperService;

    public PaperController(PaperService paperService) {
        this.paperService = paperService;
    }

    @PostMapping
    public ApiResponse<PaperResponse> create(@Valid @RequestBody PaperCreateRequest request) {
        return ApiResponse.ok(paperService.create(request));
    }

    @GetMapping("/{id}")
    public ApiResponse<PaperResponse> get(@PathVariable long id) {
        return ApiResponse.ok(paperService.get(id));
    }

    @GetMapping
    public ApiResponse<List<PaperResponse>> list(@RequestParam(required = false) String title) {
        return ApiResponse.ok(paperService.list(title));
    }

    @GetMapping("/page")
    public ApiResponse<PageResponse<PaperResponse>> page(
        @RequestParam(required = false) String title,
        @RequestParam(required = false) PaperStatus status,
        @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME)
        Instant createdFrom,
        @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME)
        Instant createdTo,
        @RequestParam(defaultValue = "1") long page,
        @RequestParam(defaultValue = "10") long size
    ) {
        return ApiResponse.ok(paperService.page(title, status, createdFrom, createdTo, page, size));
    }

    @PutMapping("/{id}")
    public ApiResponse<PaperResponse> update(
        @PathVariable long id,
        @Valid @RequestBody PaperUpdateRequest request
    ) {
        return ApiResponse.ok(paperService.update(id, request));
    }

    @DeleteMapping("/{id}")
    public ApiResponse<Map<String, Boolean>> delete(@PathVariable long id) {
        paperService.delete(id);
        return ApiResponse.ok(Map.of("deleted", true));
    }
}
