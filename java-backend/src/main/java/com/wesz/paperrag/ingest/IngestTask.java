package com.wesz.paperrag.ingest;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import java.time.Instant;

@TableName("ingest_tasks")
public class IngestTask {

    @TableId(type = IdType.AUTO)
    private Long id;
    private Long paperId;
    private Long tenantId;
    private String bizKey;
    private IngestStatus status;
    private String errorMessage;
    private Instant createdAt;
    private Instant updatedAt;

    public static IngestTask pending(Long paperId, Long tenantId, String bizKey) {
        Instant now = Instant.now();
        IngestTask task = new IngestTask();
        task.paperId = paperId;
        task.tenantId = tenantId;
        task.bizKey = bizKey;
        task.status = IngestStatus.PENDING;
        task.createdAt = now;
        task.updatedAt = now;
        return task;
    }

    public void markParsing(Instant now) {
        status = IngestStatus.PARSING;
        errorMessage = null;
        updatedAt = now;
    }

    public void markPending(Instant now) {
        status = IngestStatus.PENDING;
        errorMessage = null;
        updatedAt = now;
    }

    public void markEmbedding(Instant now) {
        status = IngestStatus.EMBEDDING;
        errorMessage = null;
        updatedAt = now;
    }

    public void markDone(Instant now) {
        status = IngestStatus.DONE;
        errorMessage = null;
        updatedAt = now;
    }

    public void markFailed(String message, Instant now) {
        status = IngestStatus.FAILED;
        errorMessage = message;
        updatedAt = now;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public Long getPaperId() {
        return paperId;
    }

    public void setPaperId(Long paperId) {
        this.paperId = paperId;
    }

    public Long getTenantId() {
        return tenantId;
    }

    public void setTenantId(Long tenantId) {
        this.tenantId = tenantId;
    }

    public String getBizKey() {
        return bizKey;
    }

    public void setBizKey(String bizKey) {
        this.bizKey = bizKey;
    }

    public IngestStatus getStatus() {
        return status;
    }

    public void setStatus(IngestStatus status) {
        this.status = status;
    }

    public String getErrorMessage() {
        return errorMessage;
    }

    public void setErrorMessage(String errorMessage) {
        this.errorMessage = errorMessage;
    }

    public Instant getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(Instant createdAt) {
        this.createdAt = createdAt;
    }

    public Instant getUpdatedAt() {
        return updatedAt;
    }

    public void setUpdatedAt(Instant updatedAt) {
        this.updatedAt = updatedAt;
    }
}
