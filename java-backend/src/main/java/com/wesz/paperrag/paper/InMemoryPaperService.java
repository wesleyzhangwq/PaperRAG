package com.wesz.paperrag.paper;

import com.wesz.paperrag.common.BusinessException;
import com.wesz.paperrag.common.PageResponse;
import java.time.Instant;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicLong;
import org.springframework.context.annotation.Profile;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;

@Service
@Profile("memory")
public class InMemoryPaperService implements PaperService {

    private final ConcurrentMap<Long, Paper> papers = new ConcurrentHashMap<>();
    private final AtomicLong idSequence = new AtomicLong(0);

    @Override
    public PaperResponse create(PaperCreateRequest request) {
        long id = idSequence.incrementAndGet();
        Paper paper = Paper.create(request.title(), request.abstractText(), request.authors());
        paper.setId(id);
        papers.put(id, paper);
        return PaperResponse.from(paper);
    }

    @Override
    public PaperResponse get(long id) {
        return PaperResponse.from(findPaper(id));
    }

    @Override
    public List<PaperResponse> list(String title) {
        String normalizedTitle = title == null ? "" : title.toLowerCase(Locale.ROOT);
        return papers.values()
            .stream()
            .filter(paper -> normalizedTitle.isBlank()
                || paper.getTitle().toLowerCase(Locale.ROOT).contains(normalizedTitle))
            .sorted(Comparator.comparing(Paper::getCreatedAt).reversed())
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
        List<PaperResponse> filtered = list(title).stream()
            .filter(paper -> status == null || paper.status() == status)
            .filter(paper -> createdFrom == null || !paper.createdAt().isBefore(createdFrom))
            .filter(paper -> createdTo == null || !paper.createdAt().isAfter(createdTo))
            .toList();
        int fromIndex = (int) Math.min((page - 1) * size, filtered.size());
        int toIndex = (int) Math.min(fromIndex + size, filtered.size());
        return new PageResponse<>(page, size, filtered.size(), filtered.subList(fromIndex, toIndex));
    }

    @Override
    public PaperResponse update(long id, PaperUpdateRequest request) {
        Paper updated = papers.compute(id, (ignored, existing) -> {
            if (existing == null) {
                throw notFound(id);
            }
            existing.update(request, Instant.now());
            return existing;
        });
        return PaperResponse.from(updated);
    }

    @Override
    public void delete(long id) {
        Paper removed = papers.remove(id);
        if (removed == null) {
            throw notFound(id);
        }
    }

    private Paper findPaper(long id) {
        Paper paper = papers.get(id);
        if (paper == null) {
            throw notFound(id);
        }
        return paper;
    }

    private BusinessException notFound(long id) {
        return new BusinessException(HttpStatus.NOT_FOUND, "Paper " + id + " not found");
    }
}
