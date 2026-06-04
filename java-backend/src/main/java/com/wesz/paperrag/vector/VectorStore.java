package com.wesz.paperrag.vector;

import java.util.List;

public interface VectorStore {

    void upsert(List<VectorDocument> documents);

    List<VectorSearchResult> search(float[] queryVector, int topK);
}
