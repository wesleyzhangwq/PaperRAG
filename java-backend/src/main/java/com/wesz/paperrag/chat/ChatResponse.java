package com.wesz.paperrag.chat;

import java.util.List;

public record ChatResponse(String answer, List<SourceResponse> sources) {
}
