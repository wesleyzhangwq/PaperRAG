package com.wesz.paperrag.eval;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import java.time.Instant;

@TableName("eval_runs")
public class EvalRun {

    @TableId(type = IdType.AUTO)
    private Long id;
    private String name;
    @TableField("ground_truth_chunk_ids")
    private String groundTruthChunkIds;
    @TableField("retrieved_chunk_ids")
    private String retrievedChunkIds;
    private Integer k;
    private Double recallAtK;
    private Double mrr;
    private Double ndcgAtK;
    private Instant createdAt;

    public static EvalRun create(EvalRunRequest request, EvalMetrics metrics) {
        EvalRun run = new EvalRun();
        run.name = request.name();
        run.groundTruthChunkIds = ids(request.groundTruthChunkIds());
        run.retrievedChunkIds = ids(request.retrievedChunkIds());
        run.k = request.k();
        run.recallAtK = metrics.recallAtK();
        run.mrr = metrics.mrr();
        run.ndcgAtK = metrics.ndcgAtK();
        run.createdAt = Instant.now();
        return run;
    }

    private static String ids(java.util.List<Long> ids) {
        if (ids == null || ids.isEmpty()) {
            return "";
        }
        return ids.stream().map(String::valueOf).collect(java.util.stream.Collectors.joining(","));
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public Integer getK() {
        return k;
    }

    public Double getRecallAtK() {
        return recallAtK;
    }

    public Double getMrr() {
        return mrr;
    }

    public Double getNdcgAtK() {
        return ndcgAtK;
    }

    public Instant getCreatedAt() {
        return createdAt;
    }
}
