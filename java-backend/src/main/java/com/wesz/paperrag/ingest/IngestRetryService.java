package com.wesz.paperrag.ingest;

import com.wesz.paperrag.common.BusinessException;
import com.wesz.paperrag.paper.Paper;
import com.wesz.paperrag.paper.PaperMapper;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;

@Service
public class IngestRetryService {

    private final IngestTaskMapper ingestTaskMapper;
    private final PaperMapper paperMapper;
    private final IngestPayloadStore payloadStore;
    private final IngestMessagePublisher publisher;

    public IngestRetryService(
        IngestTaskMapper ingestTaskMapper,
        PaperMapper paperMapper,
        IngestPayloadStore payloadStore,
        IngestMessagePublisher publisher
    ) {
        this.ingestTaskMapper = ingestTaskMapper;
        this.paperMapper = paperMapper;
        this.payloadStore = payloadStore;
        this.publisher = publisher;
    }

    public IngestRetryResponse retry(long taskId) {
        IngestTask task = ingestTaskMapper.selectById(taskId);
        if (task == null) {
            throw new BusinessException(HttpStatus.NOT_FOUND, "Ingest task " + taskId + " not found");
        }
        Paper paper = paperMapper.selectById(task.getPaperId());
        if (paper == null) {
            throw new BusinessException(HttpStatus.NOT_FOUND, "Paper " + task.getPaperId() + " not found");
        }

        task.markPending(Instant.now());
        ingestTaskMapper.updateById(task);

        IngestPayload payload = payloadStore
            .get(taskId)
            .orElseGet(() -> fallbackPayload(paper));
        publisher.publish(new IngestMessage(
            task.getId(),
            task.getPaperId(),
            task.getTenantId(),
            payload.filename(),
            payload.bytes(),
            1
        ));
        return new IngestRetryResponse(taskId, IngestStatus.PENDING, true);
    }

    private IngestPayload fallbackPayload(Paper paper) {
        String text = (paper.getTitle() == null ? "" : paper.getTitle()) + "\n\n"
            + (paper.getAbstractText() == null ? "" : paper.getAbstractText());
        return new IngestPayload(
            "paper-" + paper.getId() + ".txt",
            text.getBytes(StandardCharsets.UTF_8)
        );
    }
}
