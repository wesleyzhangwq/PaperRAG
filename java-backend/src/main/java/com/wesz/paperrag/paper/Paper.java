package com.wesz.paperrag.paper;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import java.time.Instant;

@TableName("papers")
public class Paper {

    @TableId(type = IdType.AUTO)
    private Long id;
    private Long tenantId;
    private String title;
    @TableField("abstract_text")
    private String abstractText;
    private String authors;
    private PaperStatus status;
    private String doi;
    private Instant createdAt;
    private Instant updatedAt;

    public static Paper create(String title, String abstractText, String authors) {
        Instant now = Instant.now();
        Paper paper = new Paper();
        paper.tenantId = 1L;
        paper.title = title;
        paper.abstractText = abstractText;
        paper.authors = authors;
        paper.status = PaperStatus.PENDING;
        paper.createdAt = now;
        paper.updatedAt = now;
        return paper;
    }

    public void update(PaperUpdateRequest request, Instant now) {
        title = request.title();
        abstractText = request.abstractText();
        authors = request.authors();
        updatedAt = now;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public Long getTenantId() {
        return tenantId;
    }

    public void setTenantId(Long tenantId) {
        this.tenantId = tenantId;
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public String getAbstractText() {
        return abstractText;
    }

    public void setAbstractText(String abstractText) {
        this.abstractText = abstractText;
    }

    public String getAuthors() {
        return authors;
    }

    public void setAuthors(String authors) {
        this.authors = authors;
    }

    public PaperStatus getStatus() {
        return status;
    }

    public void setStatus(PaperStatus status) {
        this.status = status;
    }

    public String getDoi() {
        return doi;
    }

    public void setDoi(String doi) {
        this.doi = doi;
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
