package com.wesz.paperrag.hybrid;

import java.util.List;

public record HybridSearchResponse(HybridMode mode, boolean degraded, List<HybridSearchResult> results) {
}
