package com.wesz.paperrag.ingest;

import com.wesz.paperrag.chunk.PaperChunk;
import com.wesz.paperrag.chunk.PaperChunkMapper;
import com.wesz.paperrag.vector.RetrievalService;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;

@Service
public class IngestWorker {

    private final IngestTaskMapper ingestTaskMapper;
    private final PaperChunkMapper paperChunkMapper;
    private final DocumentParser documentParser;
    private final DocumentChunker documentChunker;
    private final RetrievalService retrievalService;

    public IngestWorker(
        IngestTaskMapper ingestTaskMapper,
        PaperChunkMapper paperChunkMapper,
        DocumentParser documentParser,
        DocumentChunker documentChunker,
        RetrievalService retrievalService
    ) {
        this.ingestTaskMapper = ingestTaskMapper;
        this.paperChunkMapper = paperChunkMapper;
        this.documentParser = documentParser;
        this.documentChunker = documentChunker;
        this.retrievalService = retrievalService;
    }

    @Async("ingestExecutor")
    public void processAsync(Long taskId, Long paperId, Long tenantId, String filename, byte[] bytes) {
        process(taskId, paperId, tenantId, filename, bytes);
    }

    public IngestStatus process(Long taskId, Long paperId, Long tenantId, String filename, byte[] bytes) {
        try {
            update(taskId, task -> task.markParsing(Instant.now()));
            String text = documentParser.parse(filename, bytes);
            List<DocumentChunk> chunks = documentChunker.chunk(text);

            update(taskId, task -> task.markEmbedding(Instant.now()));
            List<PaperChunk> insertedChunks = new ArrayList<>();
            for (DocumentChunk chunk : chunks) {
                PaperChunk paperChunk = PaperChunk.create(
                    paperId,
                    tenantId,
                    chunk.chunkIndex(),
                    chunk.content()
                );
                paperChunkMapper.insert(paperChunk);
                insertedChunks.add(paperChunk);
            }
            retrievalService.indexChunks(insertedChunks);
            update(taskId, task -> task.markDone(Instant.now()));
            return IngestStatus.DONE;
        } catch (RuntimeException exception) {
            update(taskId, task -> task.markFailed(exception.getMessage(), Instant.now()));
            return IngestStatus.FAILED;
        }
    }

    private void update(Long taskId, TaskMutation mutation) {
        IngestTask task = ingestTaskMapper.selectById(taskId);
        if (task == null) {
            return;
        }
        mutation.apply(task);
        ingestTaskMapper.updateById(task);
    }

    private interface TaskMutation {
        void apply(IngestTask task);
    }
}
