package com.wesz.paperrag.chat;

import java.util.List;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;

@Component
@ConditionalOnProperty(name = "app.llm.provider", havingValue = "deterministic", matchIfMissing = true)
public class DeterministicLlmProvider implements LlmProvider {

    @Override
    public String answer(String question, List<SourceResponse> sources) {
        if (sources.isEmpty()) {
            return "I could not find relevant Cite Scope context for: " + question;
        }
        SourceResponse first = sources.getFirst();
        return "Based on the retrieved Cite Scope chunks, " + first.content()
            + " [chunk:" + first.chunkId() + "]";
    }
}
