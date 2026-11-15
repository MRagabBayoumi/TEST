import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app
from app.core.security import sanitize_filename, validate_pdf_content
from app.document_processing.extractor import PDFTextExtractor
from tests.create_sample_tender import generate_sample_pdf

client = TestClient(app)


def test_sanitize_filename():
    # Test path traversal prevention
    assert "tender.pdf" in sanitize_filename("../../tender.pdf")
    # Test Arabic characters preservation
    arabic_name = "كراسة_شروط_تغذية.pdf"
    assert sanitize_filename(arabic_name) == arabic_name
    # Test missing extension
    assert sanitize_filename("tender").endswith(".pdf")


def test_pdf_validation_security():
    # Invalid extension
    valid, msg = validate_pdf_content(b"%PDF-1.4 fake", "test.exe")
    assert not valid
    assert "PDF فقط" in msg

    # Invalid header
    valid, msg = validate_pdf_content(b"FAKE_HEADER", "test.pdf")
    assert not valid
    assert "توقيع الهيدر" in msg

    # Empty file
    valid, msg = validate_pdf_content(b"", "test.pdf")
    assert not valid
    assert "الملف فارغ" in msg


def test_text_extraction_and_page_preservation():
    sample_pdf = generate_sample_pdf()
    extractor = PDFTextExtractor(
        pdf_path=sample_pdf,
        document_id="doc_test_123",
        original_filename="sample_tender.pdf",
    )
    result = extractor.extract_all(save_json=True)

    # 1. Total page count check
    assert result.summary.total_pages == 4
    assert len(result.pages) == 4

    # 2. Page 1 Content Check (General Info)
    p1 = result.pages[0]
    assert p1.page_number == 1
    assert "كراسة الشروط" in p1.text
    assert "المناقصة" in p1.text
    assert "14" in p1.text
    assert not p1.is_scanned_candidate

    # 3. Page 2 Content Check (Catering Specs - Cheese 65g)
    p2 = result.pages[1]
    assert p2.page_number == 2
    assert "جبن أبيض" in p2.text
    assert "65" in p2.text
    assert not p2.is_scanned_candidate

    # 4. Page 3 Content Check (Penalties)
    p3 = result.pages[2]
    assert p3.page_number == 3
    assert "غرامة تأخير" in p3.text

    # 5. Page 4 Check (Scanned candidate processed via OCR)
    p4 = result.pages[3]
    assert p4.page_number == 4
    assert p4.is_scanned_candidate
    assert p4.extraction_method == "ocr"
    assert p4.char_count > 0


def test_fastapi_upload_endpoint():
    sample_pdf = generate_sample_pdf()
    with open(sample_pdf, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("كراسة_مناقصة_تجريبية.pdf", file_bytes, "application/pdf")},
    )

    assert response.status_code == 201
    data = response.json()
    assert "summary" in data
    assert "pages" in data
    assert data["summary"]["total_pages"] == 4

    doc_id = data["summary"]["document_id"]

    # Test single page retrieval
    page_resp = client.get(f"/api/v1/documents/{doc_id}/pages/2")
    assert page_resp.status_code == 200
    assert "جبن أبيض" in page_resp.json()["text"]

    # Test duplicate upload cache hit
    dup_response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("كراسة_مناقصة_تجريبية.pdf", file_bytes, "application/pdf")},
    )
    assert dup_response.status_code == 201
    dup_data = dup_response.json()
    assert dup_data["summary"]["document_id"] == doc_id
    assert dup_data["summary"]["total_pages"] == 4
    page_data = page_resp.json()
    assert page_data["page_number"] == 2
    assert "جبن أبيض" in page_data["text"]
