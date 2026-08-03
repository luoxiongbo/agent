import json
import tempfile
import unittest
from pathlib import Path

import pymupdf

from rag_pdf_parser import PDFParser, ParserConfig, RAGChunker, save_outputs


class ParserSmokeTest(unittest.TestCase):
    def test_native_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            pdf = temp_path / "sample.pdf"

            doc = pymupdf.open()
            for page_number in range(1, 4):
                page = doc.new_page()
                page.insert_text((72, 40), "Repeated Header", fontsize=9)
                page.insert_text((72, 100), f"{page_number}. Heading", fontsize=18)
                page.insert_text(
                    (72, 150),
                    f"This is page {page_number}. The parser should extract this text.",
                    fontsize=12,
                )
                page.insert_text((72, 800), f"Page {page_number}", fontsize=9)
            doc.save(pdf)
            doc.close()

            config = ParserConfig(
                ocr_mode="never",
                extract_tables=False,
                chunk_size_tokens=80,
                chunk_overlap_tokens=10,
            )
            parsed = PDFParser(config).parse(pdf)
            chunks = RAGChunker(config).chunk(parsed)
            outputs = save_outputs(parsed, chunks, temp_path / "output")

            self.assertEqual(parsed.page_count, 3)
            self.assertGreaterEqual(len(chunks), 1)
            self.assertTrue(outputs["document_json"].exists())
            self.assertTrue(outputs["chunks_jsonl"].exists())

            payload = json.loads(outputs["document_json"].read_text(encoding="utf-8"))
            self.assertEqual(payload["page_count"], 3)
            self.assertNotIn("Repeated Header", payload["pages"][0]["text"])


if __name__ == "__main__":
    unittest.main()
