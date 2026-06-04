package com.wesz.paperrag.eval;

import com.wesz.paperrag.common.BusinessException;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;

@Service
public class EvalService {

    private final EvalRunMapper evalRunMapper;

    public EvalService(EvalRunMapper evalRunMapper) {
        this.evalRunMapper = evalRunMapper;
    }

    public EvalRunResponse run(EvalRunRequest request) {
        EvalMetrics metrics = EvalMetrics.calculate(
            request.groundTruthChunkIds(),
            request.retrievedChunkIds(),
            request.resolvedK()
        );
        EvalRun run = EvalRun.create(
            new EvalRunRequest(
                request.name(),
                request.groundTruthChunkIds(),
                request.retrievedChunkIds(),
                request.resolvedK()
            ),
            metrics
        );
        evalRunMapper.insert(run);
        return EvalRunResponse.from(run);
    }

    public EvalRunResponse get(long runId) {
        EvalRun run = evalRunMapper.selectById(runId);
        if (run == null) {
            throw new BusinessException(HttpStatus.NOT_FOUND, "Eval run " + runId + " not found");
        }
        return EvalRunResponse.from(run);
    }
}
