package com.wesz.paperrag.ingest;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDPage;
import org.apache.pdfbox.pdmodel.PDPageContentStream;
import org.apache.pdfbox.pdmodel.font.PDType1Font;
import org.apache.pdfbox.pdmodel.font.Standard14Fonts;
import org.junit.jupiter.api.Test;

class DocumentParserTest {

    @Test
    void parsesUtf8TextBytesAndNormalizesWhitespace() {
        DocumentParser parser = new SimpleDocumentParser();

        String text = parser.parse(
            "paper.txt",
            "Paragraph one.\n\nParagraph two with  spaces.".getBytes(StandardCharsets.UTF_8)
        );

        assertThat(text).isEqualTo("Paragraph one.\n\nParagraph two with spaces.");
    }

    @Test
    void parsesPdfBytesWithPdfBox() throws Exception {
        DocumentParser parser = new SimpleDocumentParser();

        String text = parser.parse("paper.pdf", pdfBytes("PDF hybrid retrieval"));

        assertThat(text).contains("PDF hybrid retrieval");
    }

    private byte[] pdfBytes(String text) throws Exception {
        try (PDDocument document = new PDDocument();
             ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            PDPage page = new PDPage();
            document.addPage(page);
            try (PDPageContentStream content = new PDPageContentStream(document, page)) {
                content.beginText();
                content.setFont(new PDType1Font(Standard14Fonts.FontName.HELVETICA), 12);
                content.newLineAtOffset(50, 700);
                content.showText(text);
                content.endText();
            }
            document.save(output);
            return output.toByteArray();
        }
    }
}
