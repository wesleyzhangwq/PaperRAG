package com.wesz.paperrag.vector;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

class HashingEmbeddingModelTest {

    @Test
    void createsDeterministicNormalizedEmbeddings() {
        EmbeddingModel model = new HashingEmbeddingModel(16);

        float[] first = model.embed("hybrid retrieval ranking");
        float[] second = model.embed("hybrid retrieval ranking");

        assertThat(first).hasSize(16);
        assertThat(first).containsExactly(second);
        assertThat(norm(first)).isBetween(0.99, 1.01);
    }

    private double norm(float[] vector) {
        double sum = 0;
        for (float value : vector) {
            sum += value * value;
        }
        return Math.sqrt(sum);
    }
}
