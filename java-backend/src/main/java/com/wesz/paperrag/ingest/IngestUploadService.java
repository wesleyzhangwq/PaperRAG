package com.wesz.paperrag.ingest;

import com.wesz.paperrag.common.BusinessException;
import com.wesz.paperrag.paper.PaperCreateRequest;
import java.io.IOException;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

@Service
public class IngestUploadService {

    private final PaperIngestService paperIngestService;
    private final IngestPayloadStore payloadStore;
    private final IngestMessagePublisher messagePublisher;

    public IngestUploadService(
        PaperIngestService paperIngestService,
        IngestPayloadStore payloadStore,
        IngestMessagePublisher messagePublisher
    ) {
        this.paperIngestService = paperIngestService;
        this.payloadStore = payloadStore;
        this.messagePublisher = messagePublisher;
    }

    public IngestUploadResponse upload(MultipartFile file, String title) {
        byte[] bytes = readBytes(file);
        String filename = file.getOriginalFilename();
        String paperTitle = title == null || title.isBlank() ? filename : title;
        String bizKey = sha256(bytes);

        PaperIngestResult result = paperIngestService.createPaperAndTask(
            new PaperCreateRequest(paperTitle, "", "Uploaded"),
            bizKey
        );
        payloadStore.put(result.taskId(), new IngestPayload(filename, bytes));
        messagePublisher.publish(new IngestMessage(
            result.taskId(),
            result.paperId(),
            1L,
            filename,
            bytes,
            0
        ));
        return new IngestUploadResponse(result.paperId(), result.taskId(), IngestStatus.PENDING);
    }

    private byte[] readBytes(MultipartFile file) {
        if (file == null || file.isEmpty()) {
            throw new BusinessException(HttpStatus.BAD_REQUEST, "Upload file must not be empty");
        }
        try {
            return file.getBytes();
        } catch (IOException exception) {
            throw new BusinessException(HttpStatus.BAD_REQUEST, "Unable to read upload file");
        }
    }

    private String sha256(byte[] bytes) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            return HexFormat.of().formatHex(digest.digest(bytes));
        } catch (NoSuchAlgorithmException exception) {
            throw new IllegalStateException("SHA-256 is not available", exception);
        }
    }
}
