package com.wesz.paperrag.paper;

import static org.assertj.core.api.Assertions.assertThat;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
class PaperPersistenceServiceTest {

    @Autowired
    private PaperPersistenceService paperPersistenceService;

    @Test
    void savesAndPagesThroughMyBatisPlusIService() {
        String title = "IService Paper " + System.nanoTime();
        Paper paper = Paper.create(title, "persistence abstraction", "Alice");

        paperPersistenceService.save(paper);

        Page<Paper> page = paperPersistenceService.page(
            Page.of(1, 10),
            new LambdaQueryWrapper<Paper>().eq(Paper::getTitle, title)
        );

        assertThat(paper.getId()).isNotNull();
        assertThat(page.getTotal()).isEqualTo(1);
        assertThat(page.getRecords().getFirst().getTitle()).isEqualTo(title);
    }
}
