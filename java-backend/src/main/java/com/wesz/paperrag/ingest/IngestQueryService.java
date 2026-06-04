package com.wesz.paperrag.ingest;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.wesz.paperrag.chunk.PaperChunk;
import com.wesz.paperrag.chunk.PaperChunkMapper;
import com.wesz.paperrag.common.BusinessException;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;

@Service
public class IngestQueryService {

    private final IngestTaskMapper ingestTaskMapper;
    private final PaperChunkMapper paperChunkMapper;

    public IngestQueryService(IngestTaskMapper ingestTaskMapper, PaperChunkMapper paperChunkMapper) {
        this.ingestTaskMapper = ingestTaskMapper;
        this.paperChunkMapper = paperChunkMapper;
    }

    public IngestTaskResponse getTask(long taskId) {
        IngestTask task = ingestTaskMapper.selectById(taskId);
        if (task == null) {
            throw new BusinessException(HttpStatus.NOT_FOUND, "Ingest task " + taskId + " not found");
        }
        Long chunkCount = paperChunkMapper.selectCount(
            new LambdaQueryWrapper<PaperChunk>().eq(PaperChunk::getPaperId, task.getPaperId())
        );
        return IngestTaskResponse.from(task, chunkCount);
    }
}
