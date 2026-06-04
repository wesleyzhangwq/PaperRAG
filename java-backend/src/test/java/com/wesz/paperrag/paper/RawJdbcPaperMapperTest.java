package com.wesz.paperrag.paper;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
class RawJdbcPaperMapperTest {

    @Autowired
    private PaperMapper paperMapper;

    @Autowired
    private RawJdbcPaperMapper rawJdbcPaperMapper;

    @Test
    void selectsPaperByIdWithRawJdbc() {
        Paper paper = Paper.create(
            "Raw JDBC Paper",
            "compare mapper styles",
            "Alice"
        );
        paperMapper.insert(paper);

        Paper found = rawJdbcPaperMapper.selectById(paper.getId()).orElseThrow();

        assertThat(found.getTitle()).isEqualTo("Raw JDBC Paper");
        assertThat(found.getAbstractText()).isEqualTo("compare mapper styles");
    }
}
