import pytest
from pathlib import Path
from fpdf import FPDF
import pymupdf
from app.document_processing.table_extractor import TableExtractor
from app.document_processing.extractor import PDFTextExtractor
from app.schemas.document import ExtractedTable


def create_pdf_with_catering_table(filepath: Path) -> Path:
    """Creates a sample PDF with a structured catering specification table."""
    pdf = FPDF()
    pdf.add_font("Arial", "", "C:/Windows/Fonts/arial.ttf")
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(text="Hospital Food Supply Table", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(5)

    pdf.set_font("Arial", size=11)
    col_widths = (40, 40, 50, 40)
    
    # Headers
    headers = ["Item", "Grams", "Frequency", "Packaging"]
    for w, h in zip(col_widths, headers):
        pdf.cell(w, 10, h, border=1)
    pdf.ln()

    # Rows
    rows = [
        ["White Cheese", "65g", "Daily 7/7", "Sealed Pack"],
        ["Grilled Meat", "200g", "3 times/week", "Thermal Box"],
        ["Basmati Rice", "250g", "Daily 7/7", "Container"],
    ]
    for row in rows:
        for w, cell in zip(col_widths, row):
            pdf.cell(w, 9, cell, border=1)
        pdf.ln()

    pdf.output(str(filepath))
    return filepath


def test_table_extractor_finds_tables(tmp_path):
    pdf_path = create_pdf_with_catering_table(tmp_path / "test_table.pdf")
    doc = pymupdf.open(pdf_path)
    page = doc[0]

    tables = TableExtractor.extract_tables_from_page(page, page_number=1)
    doc.close()

    assert len(tables) >= 1
    tbl = tables[0]

    assert isinstance(tbl, ExtractedTable)
    assert tbl.page_number == 1
    assert tbl.table_id == "tbl_p1_1"
    assert tbl.col_count == 4
    assert tbl.row_count >= 3
    assert "Item" in tbl.headers
    assert "Grams" in tbl.headers

    # Check Markdown representation
    assert "| Item | Grams |" in tbl.markdown
    assert "| --- |" in tbl.markdown

    # Check data_rows dictionary conversion for ERP integration
    assert len(tbl.data_rows) >= 2
    row1 = tbl.data_rows[0]
    assert row1["Item"] == "White Cheese"
    assert row1["Grams"] == "65g"


def test_pdf_extractor_integrates_tables(tmp_path):
    pdf_path = create_pdf_with_catering_table(tmp_path / "test_table_doc.pdf")
    extractor = PDFTextExtractor(
        pdf_path=pdf_path,
        document_id="doc_tbl_123",
        original_filename="test_table_doc.pdf",
    )
    result = extractor.extract_all(save_json=True)

    assert result.summary.total_tables_count >= 1
    page1 = result.pages[0]
    assert len(page1.tables) >= 1
    assert page1.tables[0].col_count == 4
