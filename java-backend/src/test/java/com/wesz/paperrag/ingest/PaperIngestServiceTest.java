package com.wesz.paperrag.ingest;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.wesz.paperrag.common.BusinessException;
import com.wesz.paperrag.paper.PaperCreateRequest;
import com.wesz.paperrag.paper.PaperMapper;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
class PaperIngestServiceTest {

    @Autowired
    private PaperIngestService paperIngestService;

    @Autowired
    private PaperMapper paperMapper;

    @Autowired
    private IngestTaskMapper ingestTaskMapper;

    @Test
    void createsPaperAndTaskInOneTransaction() {
        String bizKey = "biz-" + System.nanoTime();

        PaperIngestResult result = paperIngestService.createPaperAndTask(
            new PaperCreateRequest("Transactional Paper", "two table write", "Alice"),
            bizKey
        );

        assertThat(paperMapper.selectById(result.paperId())).isNotNull();
        assertThat(ingestTaskMapper.selectById(result.taskId())).isNotNull();
    }

    @Test
    void duplicateBizKeyRollsBackPaperInsert() {
        String bizKey = "duplicate-" + System.nanoTime();
        long before = paperMapper.selectCount(null);

        paperIngestService.createPaperAndTask(
            new PaperCreateRequest("Original Paper", "first", "Alice"),
            bizKey
        );

        assertThatThrownBy(() -> paperIngestService.createPaperAndTask(
            new PaperCreateRequest("Duplicate Paper", "second", "Bob"),
            bizKey
        )).isInstanceOf(BusinessException.class)
            .hasMessageContaining("already exists");

        assertThat(paperMapper.selectCount(null)).isEqualTo(before + 1);
    }
}
