package com.wesz.paperrag.vector;

public interface EmbeddingModel {

    float[] embed(String text);
}
