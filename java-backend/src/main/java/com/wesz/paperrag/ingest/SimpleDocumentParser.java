package com.wesz.paperrag.ingest;

import com.wesz.paperrag.common.BusinessException;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import org.apache.pdfbox.Loader;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.text.PDFTextStripper;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Component;

@Component
public class SimpleDocumentParser implements DocumentParser {

    @Override
    public String parse(String filename, byte[] bytes) {
        String raw = isPdf(filename) ? parsePdf(bytes) : new String(bytes, StandardCharsets.UTF_8);
        return normalize(raw);
    }

    private boolean isPdf(String filename) {
        return filename != null && filename.toLowerCase().endsWith(".pdf");
    }

    private String parsePdf(byte[] bytes) {
        try (PDDocument document = Loader.loadPDF(bytes)) {
            return new PDFTextStripper().getText(document);
        } catch (IOException exception) {
            throw new BusinessException(HttpStatus.BAD_REQUEST, "Unable to parse PDF upload");
        }
    }

    private String normalize(String raw) {
        return raw
            .replace("\r\n", "\n")
            .replace("\r", "\n")
            .replaceAll("[ \\t]+", " ")
            .replaceAll("\\n{3,}", "\n\n")
            .trim();
    }
}
