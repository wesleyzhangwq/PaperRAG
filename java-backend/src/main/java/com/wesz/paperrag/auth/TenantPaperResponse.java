package com.wesz.paperrag.auth;

import com.wesz.paperrag.paper.Paper;

public record TenantPaperResponse(Long id, Long tenantId, String title) {

    static TenantPaperResponse from(Paper paper) {
        return new TenantPaperResponse(paper.getId(), paper.getTenantId(), paper.getTitle());
    }
}
