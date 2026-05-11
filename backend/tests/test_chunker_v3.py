"""Regression tests for chunk_pages_v3 title/next-block merge behavior."""
from __future__ import annotations

import unittest

from app.utils.chunker import chunk_pages_v3


class TestChunkPagesV3TitleMerge(unittest.TestCase):
    def test_title_merge_does_not_rechunk_next_block_twice(self) -> None:
        """When a heading is merged with the following block, that block must not be processed again."""
        marker = "XYZZY_UNIQUE_MARKER"
        page_text = f"Methods\n\n{marker}. " + "paddingword " * 40
        chunks = chunk_pages_v3([(1, page_text)])
        hits = sum(c.text.count(marker) for c in chunks)
        self.assertEqual(
            hits,
            1,
            "Duplicate block processing would emit the marker from multiple chunk passes.",
        )


if __name__ == "__main__":
    unittest.main()
