import pytest
from app.document_processing.semantic_chunker import ArabicSemanticChunker
from app.schemas.document import ExtractedTable, PageContent


def test_arabic_semantic_chunker_basic_text():
    chunker = ArabicSemanticChunker(target_chunk_size=100, chunk_overlap=30)
    page = PageContent(
        page_number=2,
        char_count=250,
        word_count=40,
        has_text=True,
        is_scanned_candidate=False,
        image_count=0,
        extraction_method="digital",
        width=595.0,
        height=842.0,
        text=(
            "أولاً: وجبة الإفطار الخاصة بالأطباء والتمريض.\n\n"
            "تشمل بيض مسلوق وجبن أبيض كامل الدسم وخبز بلدي طازج.\n\n"
            "ثانياً: وجبة الغداء للمرضى المقيمين بالأقسام الداخلية.\n\n"
            "تشمل لحم كندوز مسلوق مع أرز وخضار مشكل طازج وسلطة خضراء."
        ),
    )

    chunks = chunker.chunk_page(page, document_id="doc_test_1", filename="tender.pdf")
    assert len(chunks) >= 2
    for chk in chunks:
        assert chk.document_id == "doc_test_1"
        assert chk.page_number == 2
        assert chk.chunk_type == "text"
        assert chk.metadata["filename"] == "tender.pdf"
        assert len(chk.content) > 0


def test_arabic_semantic_chunker_keeps_tables_intact():
    chunker = ArabicSemanticChunker(target_chunk_size=100)
    table = ExtractedTable(
        table_id="tbl_p1_1",
        page_number=1,
        row_count=3,
        col_count=2,
        headers=["الصنف", "الكمية"],
        rows=[["لحم بلدي", "150 جرام"], ["أرز", "200 جرام"]],
        markdown="| الصنف | الكمية |\n| --- | --- |\n| لحم بلدي | 150 جرام |\n| أرز | 200 جرام |",
        data_rows=[{"الصنف": "لحم بلدي", "الكمية": "150 جرام"}],
    )

    page = PageContent(
        page_number=1,
        char_count=50,
        word_count=10,
        has_text=True,
        is_scanned_candidate=False,
        image_count=0,
        extraction_method="digital",
        width=595.0,
        height=842.0,
        text="جدول توريد الأغذية لمستشفى المنيرة العام",
        tables=[table],
    )

    chunks = chunker.chunk_page(page, document_id="doc_table_test", filename="specs.pdf")

    table_chunks = [c for c in chunks if c.chunk_type == "table"]
    assert len(table_chunks) == 1

    tbl_chk = table_chunks[0]
    assert tbl_chk.page_number == 1
    assert "لحم بلدي" in tbl_chk.content
    assert "150 جرام" in tbl_chk.content
    assert tbl_chk.metadata["table_id"] == "tbl_p1_1"
    assert tbl_chk.metadata["headers"] == ["الصنف", "الكمية"]
