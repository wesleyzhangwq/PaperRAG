package com.wesz.paperrag.ingest;

import com.wesz.paperrag.common.BusinessException;
import com.wesz.paperrag.paper.Paper;
import com.wesz.paperrag.paper.PaperCreateRequest;
import com.wesz.paperrag.paper.PaperMapper;
import org.springframework.dao.DuplicateKeyException;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
public class PaperIngestService {

    private final PaperMapper paperMapper;
    private final IngestTaskMapper ingestTaskMapper;

    public PaperIngestService(PaperMapper paperMapper, IngestTaskMapper ingestTaskMapper) {
        this.paperMapper = paperMapper;
        this.ingestTaskMapper = ingestTaskMapper;
    }

    @Transactional
    public PaperIngestResult createPaperAndTask(PaperCreateRequest request, String bizKey) {
        Paper paper = Paper.create(request.title(), request.abstractText(), request.authors());
        paperMapper.insert(paper);

        try {
            IngestTask task = IngestTask.pending(paper.getId(), paper.getTenantId(), bizKey);
            ingestTaskMapper.insert(task);
            return new PaperIngestResult(paper.getId(), task.getId());
        } catch (DuplicateKeyException exception) {
            throw new BusinessException(
                HttpStatus.CONFLICT,
                "Ingest task already exists for bizKey " + bizKey
            );
        }
    }
}
