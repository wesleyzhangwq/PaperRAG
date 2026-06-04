package com.wesz.paperrag.paper;

import com.wesz.paperrag.common.PageResponse;
import java.time.Instant;
import java.util.List;

public interface PaperService {

    PaperResponse create(PaperCreateRequest request);

    PaperResponse get(long id);

    List<PaperResponse> list(String title);

    PageResponse<PaperResponse> page(
        String title,
        PaperStatus status,
        Instant createdFrom,
        Instant createdTo,
        long page,
        long size
    );

    PaperResponse update(long id, PaperUpdateRequest request);

    void delete(long id);
}
