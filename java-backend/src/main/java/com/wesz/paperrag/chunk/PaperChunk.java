package com.wesz.paperrag.chunk;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import java.time.Instant;

@TableName("paper_chunks")
public class PaperChunk {

    @TableId(type = IdType.AUTO)
    private Long id;
    private Long paperId;
    private Long tenantId;
    private Integer chunkIndex;
    private String content;
    private Integer tokenCount;
    private Instant createdAt;

    public static PaperChunk create(Long paperId, Long tenantId, int chunkIndex, String content) {
        PaperChunk chunk = new PaperChunk();
        chunk.paperId = paperId;
        chunk.tenantId = tenantId;
        chunk.chunkIndex = chunkIndex;
        chunk.content = content;
        chunk.tokenCount = estimateTokenCount(content);
        chunk.createdAt = Instant.now();
        return chunk;
    }

    private static int estimateTokenCount(String content) {
        if (content == null || content.isBlank()) {
            return 0;
        }
        return content.trim().split("\\s+").length;
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

    public Integer getChunkIndex() {
        return chunkIndex;
    }

    public void setChunkIndex(Integer chunkIndex) {
        this.chunkIndex = chunkIndex;
    }

    public String getContent() {
        return content;
    }

    public void setContent(String content) {
        this.content = content;
    }

    public Integer getTokenCount() {
        return tokenCount;
    }

    public void setTokenCount(Integer tokenCount) {
        this.tokenCount = tokenCount;
    }

    public Instant getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(Instant createdAt) {
        this.createdAt = createdAt;
    }
}
