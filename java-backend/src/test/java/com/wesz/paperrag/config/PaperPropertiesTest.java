package com.wesz.paperrag.config;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
class PaperPropertiesTest {

    @Autowired
    private PaperProperties paperProperties;

    @Test
    void bindsMaxUploadSizeFromApplicationYaml() {
        assertThat(paperProperties.maxUploadMb()).isEqualTo(50);
    }
}
