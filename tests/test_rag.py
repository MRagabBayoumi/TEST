import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from app.main import app
from app.rag.qa_engine import TenderRAGService
from app.schemas.chunk import DocumentChunk
from app.schemas.rag import RAGQuestionRequest
from app.vector_store.qdrant_service import QdrantVectorStore

client = TestClient(app)


def test_rag_service_with_in_memory_qdrant():
    q_client = QdrantClient(location=":memory:")
    store = QdrantVectorStore(client=q_client, collection_name="test_rag_collection")

    # Add sample catering chunks
    chunks = [
        DocumentChunk(
            chunk_id="chk_c1",
            document_id="doc_rag_1",
            page_number=2,
            chunk_type="text",
            content="المواصفات الفنية: وجبة الإفطار تشمل جبن أبيض 65 جرام كامل الدسم مغلف آلياً.",
            char_count=70,
        ),
        DocumentChunk(
            chunk_id="chk_c2",
            document_id="doc_rag_1",
            page_number=3,
            chunk_type="text",
            content="الجزاءات: تطبق غرامة تأخير قدرها 1% عن كل ساعة تأخير في تسليم الوجبات.",
            char_count=65,
        ),
        DocumentChunk(
            chunk_id="chk_c3",
            document_id="doc_rag_1",
            page_number=16,
            chunk_type="table",
            content="| الوجبة | الصنف | الكمية | التكرار |\n| --- | --- | --- | --- |\n| فطار | جبن أبيض | 65 جم | 7/7 |",
            char_count=85,
        ),
    ]

    store.index_chunks(chunks)

    service = TenderRAGService(vector_store=store)

    req = RAGQuestionRequest(
        question="ما هي كمية الجبن المقررة في وجبة الإفطار؟",
        document_id="doc_rag_1",
        top_k=3,
    )

    res = service.ask(req)

    assert res.question == req.question
    assert len(res.citations) >= 1
    assert any(c.page_number in (2, 16) for c in res.citations)
    assert len(res.answer) > 20
    assert "جبن" in res.answer or "65" in res.answer


def test_rag_api_endpoint():
    response = client.post(
        "/api/v1/rag/ask",
        json={"question": "ما هي شروط توريد الأغذية؟", "top_k": 2},
    )
    assert response.status_code == 200
    data = response.json()
    assert "question" in data
    assert "answer" in data
    assert "citations" in data
