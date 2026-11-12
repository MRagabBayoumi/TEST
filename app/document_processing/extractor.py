import json
from pathlib import Path
from typing import Optional
import pymupdf
from app.core.config import settings
from app.document_processing.cleaner import clean_arabic_text
from app.ocr.base import BaseOCREngine
from app.ocr.tesseract_engine import TesseractOCREngine
from app.schemas.document import DocumentExtractionResult, DocumentSummary, PageContent


class PDFTextExtractor:
    """
    Extracts digital and OCR text page-by-page while preserving exact page numbers
    and document structure. Automatically routes image-only/scanned pages through OCR.
    """

    def __init__(
        self,
        pdf_path: Path,
        document_id: str,
        original_filename: str,
        ocr_engine: Optional[BaseOCREngine] = None,
    ):
        self.pdf_path = Path(pdf_path)
        self.document_id = document_id
        self.original_filename = original_filename
        self.ocr_engine = ocr_engine or TesseractOCREngine()

        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file does not exist: {self.pdf_path}")

    def extract_all(self, save_json: bool = True) -> DocumentExtractionResult:
        """
        Extracts all pages from the PDF document and compiles structured results.
        Automatically runs OCR when scanned pages are detected.
        """
        doc = pymupdf.open(self.pdf_path)
        total_pages = len(doc)
        pages: list[PageContent] = []
        digital_pages_count = 0
        scanned_pages_count = 0
        ocr_pages_count = 0

        for page_index in range(total_pages):
            page = doc.load_page(page_index)
            page_number = page_index + 1  # 1-indexed

            # Extract raw digital text
            raw_text = page.get_text("text")
            cleaned_text = clean_arabic_text(raw_text)

            char_count = len(cleaned_text)
            word_count = len(cleaned_text.split()) if cleaned_text else 0

            # Embedded images detection
            image_list = page.get_images(full=True)
            image_count = len(image_list)

            # Detect whether this page lacks text and needs OCR
            is_scanned = (char_count < 40 and image_count > 0) or (char_count == 0)

            extraction_method = "digital"
            ocr_confidence = None

            if is_scanned:
                scanned_pages_count += 1
                # Run OCR if available
                if self.ocr_engine and self.ocr_engine.is_available():
                    ocr_raw_text, confidence = self.ocr_engine.extract_from_page(
                        page, dpi=settings.OCR_DPI
                    )
                    cleaned_ocr = clean_arabic_text(ocr_raw_text)

                    if cleaned_ocr:
                        cleaned_text = cleaned_ocr
                        char_count = len(cleaned_text)
                        word_count = len(cleaned_text.split())
                        extraction_method = "ocr"
                        ocr_confidence = confidence
                        ocr_pages_count += 1
            else:
                digital_pages_count += 1

            rect = page.rect
            pages.append(
                PageContent(
                    page_number=page_number,
                    char_count=char_count,
                    word_count=word_count,
                    has_text=char_count > 0,
                    is_scanned_candidate=is_scanned,
                    image_count=image_count,
                    extraction_method=extraction_method,
                    ocr_confidence=ocr_confidence,
                    width=float(rect.width),
                    height=float(rect.height),
                    text=cleaned_text,
                )
            )

        doc.close()

        file_size_bytes = self.pdf_path.stat().st_size
        summary = DocumentSummary(
            document_id=self.document_id,
            filename=self.original_filename,
            file_size_bytes=file_size_bytes,
            total_pages=total_pages,
            digital_pages_count=digital_pages_count,
            scanned_pages_count=scanned_pages_count,
            ocr_pages_count=ocr_pages_count,
        )

        json_path_str = None
        if save_json:
            output_file = settings.PROCESSED_DIR / f"{self.document_id}.json"
            result_dict = {
                "summary": summary.model_dump(mode="json"),
                "pages": [p.model_dump(mode="json") for p in pages],
            }
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result_dict, f, ensure_ascii=False, indent=2)
            json_path_str = str(output_file)

        return DocumentExtractionResult(
            summary=summary, pages=pages, processed_json_path=json_path_str
        )
