package com.wesz.paperrag.paper;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.wesz.paperrag.common.BusinessException;
import com.wesz.paperrag.common.PageResponse;
import java.time.Instant;
import java.util.List;
import java.util.Locale;
import org.springframework.context.annotation.Profile;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;

@Service
@Profile("!memory")
public class MyBatisPaperService implements PaperService {

    private final PaperPersistenceService paperPersistenceService;

    public MyBatisPaperService(PaperPersistenceService paperPersistenceService) {
        this.paperPersistenceService = paperPersistenceService;
    }

    @Override
    public PaperResponse create(PaperCreateRequest request) {
        Paper paper = Paper.create(request.title(), request.abstractText(), request.authors());
        paperPersistenceService.save(paper);
        return PaperResponse.from(paper);
    }

    @Override
    public PaperResponse get(long id) {
        return PaperResponse.from(findPaper(id));
    }

    @Override
    public List<PaperResponse> list(String title) {
        return paperPersistenceService.list(query(title, null, null, null))
            .stream()
            .map(PaperResponse::from)
            .toList();
    }

    @Override
    public PageResponse<PaperResponse> page(
        String title,
        PaperStatus status,
        Instant createdFrom,
        Instant createdTo,
        long page,
        long size
    ) {
        Page<Paper> result = paperPersistenceService.page(
            Page.of(page, size),
            query(title, status, createdFrom, createdTo)
        );
        List<PaperResponse> items = result.getRecords()
            .stream()
            .map(PaperResponse::from)
            .toList();
        return new PageResponse<>(result.getCurrent(), result.getSize(), result.getTotal(), items);
    }

    @Override
    public PaperResponse update(long id, PaperUpdateRequest request) {
        Paper paper = findPaper(id);
        paper.update(request, Instant.now());
        paperPersistenceService.updateById(paper);
        return PaperResponse.from(paper);
    }

    @Override
    public void delete(long id) {
        boolean deleted = paperPersistenceService.removeById(id);
        if (!deleted) {
            throw notFound(id);
        }
    }

    private Paper findPaper(long id) {
        Paper paper = paperPersistenceService.getById(id);
        if (paper == null) {
            throw notFound(id);
        }
        return paper;
    }

    private LambdaQueryWrapper<Paper> query(
        String title,
        PaperStatus status,
        Instant createdFrom,
        Instant createdTo
    ) {
        LambdaQueryWrapper<Paper> wrapper = new LambdaQueryWrapper<>();
        if (title != null && !title.isBlank()) {
            wrapper.apply(
                "LOWER(title) LIKE {0}",
                "%" + title.toLowerCase(Locale.ROOT) + "%"
            );
        }
        if (status != null) {
            wrapper.eq(Paper::getStatus, status);
        }
        if (createdFrom != null) {
            wrapper.ge(Paper::getCreatedAt, createdFrom);
        }
        if (createdTo != null) {
            wrapper.le(Paper::getCreatedAt, createdTo);
        }
        wrapper.orderByDesc(Paper::getCreatedAt);
        return wrapper;
    }

    private BusinessException notFound(long id) {
        return new BusinessException(HttpStatus.NOT_FOUND, "Paper " + id + " not found");
    }
}
