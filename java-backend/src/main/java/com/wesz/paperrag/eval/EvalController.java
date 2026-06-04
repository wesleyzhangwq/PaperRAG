package com.wesz.paperrag.eval;

import com.wesz.paperrag.common.ApiResponse;
import jakarta.validation.Valid;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/eval")
public class EvalController {

    private final EvalService evalService;

    public EvalController(EvalService evalService) {
        this.evalService = evalService;
    }

    @PostMapping("/run")
    public ApiResponse<EvalRunResponse> run(@Valid @RequestBody EvalRunRequest request) {
        return ApiResponse.ok(evalService.run(request));
    }

    @GetMapping("/runs/{runId}")
    public ApiResponse<EvalRunResponse> get(@PathVariable long runId) {
        return ApiResponse.ok(evalService.get(runId));
    }
}
