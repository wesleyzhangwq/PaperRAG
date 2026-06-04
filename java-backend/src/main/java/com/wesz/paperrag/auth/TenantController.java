package com.wesz.paperrag.auth;

import com.wesz.paperrag.common.ApiResponse;
import com.wesz.paperrag.common.BusinessException;
import com.wesz.paperrag.paper.Paper;
import com.wesz.paperrag.paper.PaperMapper;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/tenant")
public class TenantController {

    private final PaperMapper paperMapper;

    public TenantController(PaperMapper paperMapper) {
        this.paperMapper = paperMapper;
    }

    @GetMapping("/me")
    public ApiResponse<TenantMeResponse> me() {
        AuthenticatedUser user = TenantContext.require();
        return ApiResponse.ok(new TenantMeResponse(user.username(), user.tenantId()));
    }

    @GetMapping("/papers/{id}")
    public ApiResponse<TenantPaperResponse> getPaper(@PathVariable long id) {
        AuthenticatedUser user = TenantContext.require();
        Paper paper = paperMapper.selectById(id);
        if (paper == null) {
            throw new BusinessException(HttpStatus.NOT_FOUND, "Paper " + id + " not found");
        }
        if (!paper.getTenantId().equals(user.tenantId())) {
            throw new BusinessException(HttpStatus.FORBIDDEN, "Paper belongs to another tenant");
        }
        return ApiResponse.ok(TenantPaperResponse.from(paper));
    }
}
