package com.wesz.paperrag.ingest;

public interface DocumentParser {

    String parse(String filename, byte[] bytes);
}
