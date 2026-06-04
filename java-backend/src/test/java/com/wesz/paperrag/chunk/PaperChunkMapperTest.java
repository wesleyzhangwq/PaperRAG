package com.wesz.paperrag.chunk;

import static org.assertj.core.api.Assertions.assertThat;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.wesz.paperrag.paper.Paper;
import com.wesz.paperrag.paper.PaperMapper;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
class PaperChunkMapperTest {

    @Autowired
    private PaperMapper paperMapper;

    @Autowired
    private PaperChunkMapper paperChunkMapper;

    @Test
    void insertsAndListsChunksByPaperIdInOrder() {
        Paper paper = Paper.create("Chunked Paper", "abstract", "Alice");
        paperMapper.insert(paper);

        paperChunkMapper.insert(PaperChunk.create(paper.getId(), paper.getTenantId(), 1, "second chunk"));
        paperChunkMapper.insert(PaperChunk.create(paper.getId(), paper.getTenantId(), 0, "first chunk"));

        var chunks = paperChunkMapper.selectList(
            new LambdaQueryWrapper<PaperChunk>()
                .eq(PaperChunk::getPaperId, paper.getId())
                .orderByAsc(PaperChunk::getChunkIndex)
        );

        assertThat(chunks).hasSize(2);
        assertThat(chunks.getFirst().getChunkIndex()).isZero();
        assertThat(chunks.getFirst().getContent()).isEqualTo("first chunk");
        assertThat(chunks.getLast().getContent()).isEqualTo("second chunk");
    }
}
