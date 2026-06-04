package com.wesz.paperrag.chat;

import java.util.List;

public interface LlmProvider {

    String answer(String question, List<SourceResponse> sources);
}
