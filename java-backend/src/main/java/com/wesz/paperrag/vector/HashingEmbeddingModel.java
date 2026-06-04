package com.wesz.paperrag.vector;

import java.util.Locale;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

@Component
public class HashingEmbeddingModel implements EmbeddingModel {

    private final int dimensions;

    public HashingEmbeddingModel(@Value("${app.vector.embedding-dimensions:128}") int dimensions) {
        this.dimensions = dimensions;
    }

    @Override
    public float[] embed(String text) {
        float[] vector = new float[dimensions];
        if (text == null || text.isBlank()) {
            return vector;
        }
        for (String token : text.toLowerCase(Locale.ROOT).trim().split("\\s+")) {
            int hash = token.hashCode();
            int index = Math.floorMod(hash, dimensions);
            vector[index] += hash < 0 ? -1.0f : 1.0f;
        }
        normalize(vector);
        return vector;
    }

    private void normalize(float[] vector) {
        double sum = 0;
        for (float value : vector) {
            sum += value * value;
        }
        if (sum == 0) {
            return;
        }
        float norm = (float) Math.sqrt(sum);
        for (int index = 0; index < vector.length; index++) {
            vector[index] = vector[index] / norm;
        }
    }
}
