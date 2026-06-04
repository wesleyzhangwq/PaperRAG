package com.wesz.paperrag.chat;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.wesz.paperrag.common.ApiResponse;
import jakarta.validation.Valid;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.mvc.method.annotation.StreamingResponseBody;

@RestController
@RequestMapping("/api/chat")
public class ChatController {

    private final ChatService chatService;
    private final ObjectMapper objectMapper;

    public ChatController(ChatService chatService, ObjectMapper objectMapper) {
        this.chatService = chatService;
        this.objectMapper = objectMapper;
    }

    @PostMapping
    public ApiResponse<ChatResponse> chat(@Valid @RequestBody ChatRequest request) {
        return ApiResponse.ok(chatService.answer(request.question()));
    }

    @PostMapping(path = "/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public ResponseEntity<StreamingResponseBody> stream(@Valid @RequestBody ChatRequest request) {
        ChatResponse response = chatService.answer(request.question());
        StreamingResponseBody body = output -> {
            writeEvent(output, "sources", objectMapper.writeValueAsString(response.sources()));
            for (String token : response.answer().split(" ")) {
                writeEvent(output, "token", objectMapper.writeValueAsString(token + " "));
            }
            writeEvent(output, "done", objectMapper.writeValueAsString("ok"));
        };
        return ResponseEntity.ok()
            .contentType(MediaType.TEXT_EVENT_STREAM)
            .body(body);
    }

    private void writeEvent(java.io.OutputStream output, String event, String data) throws IOException {
        output.write(("event: " + event + "\n").getBytes(StandardCharsets.UTF_8));
        output.write(("data: " + data + "\n\n").getBytes(StandardCharsets.UTF_8));
        output.flush();
    }
}
