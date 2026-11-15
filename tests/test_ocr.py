import pytest
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display
from fastapi.testclient import TestClient
from app.main import app
from app.ocr.tesseract_engine import TesseractOCREngine
from app.document_processing.extractor import PDFTextExtractor
from tests.create_sample_tender import generate_sample_pdf

client = TestClient(app)


def test_tesseract_engine_availability():
    engine = TesseractOCREngine()
    assert engine.is_available() is True


def test_ocr_extraction_on_image():
    engine = TesseractOCREngine()
    img = Image.new("RGB", (700, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 26)

    reshaped_text = get_display(arabic_reshaper.reshape("توريد الوجبات المطهية للمستشفيات"))
    draw.text((50, 70), reshaped_text, fill=(0, 0, 0), font=font)

    text, confidence = engine.extract_from_image(img)
    assert confidence > 0.6
    assert len(text) > 0


def test_pdf_scanned_page_ocr_pipeline():
    sample_pdf = generate_sample_pdf()
    extractor = PDFTextExtractor(
        pdf_path=sample_pdf,
        document_id="doc_ocr_test",
        original_filename="sample_tender.pdf",
    )
    result = extractor.extract_all(save_json=True)

    # Verify Summary counts
    assert result.summary.total_pages == 4
    assert result.summary.digital_pages_count == 3
    assert result.summary.scanned_pages_count == 1
    assert result.summary.ocr_pages_count == 1

    # Verify Digital Pages (Pages 1, 2, 3)
    assert result.pages[0].extraction_method == "digital"
    assert result.pages[1].extraction_method == "digital"
    assert result.pages[2].extraction_method == "digital"

    # Verify Scanned Page 4 was OCR-processed
    p4 = result.pages[3]
    assert p4.page_number == 4
    assert p4.is_scanned_candidate is True
    assert p4.extraction_method == "ocr"
    assert p4.ocr_confidence is not None
    assert p4.ocr_confidence > 0.50
    assert p4.has_text is True
    assert "استبعاد" in p4.text or "شهادة صحية" in p4.text or "جامعة القاهرة" in p4.text


def test_ocr_error_handling():
    engine = TesseractOCREngine()
    # Test blank white image
    blank_img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    text, conf = engine.extract_from_image(blank_img)
    assert text == ""
    assert conf == 0.0
