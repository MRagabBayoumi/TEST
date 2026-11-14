import numpy as np
import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from app.main import app
from app.schemas.chunk import DocumentChunk
from app.vector_store.embeddings import LocalSemanticEmbeddingEngine
from app.vector_store.qdrant_service import QdrantVectorStore

client = TestClient(app)


def test_local_semantic_embedding_engine():
    engine = LocalSemanticEmbeddingEngine()
    assert engine.dimension == 384

    v1 = np.array(engine.embed_text("توريد وجبات اللحوم لمستشفيات جامعة القاهرة"))
    v2 = np.array(engine.embed_text("لحوم مطهية طازجة للمستشفى"))
    v_unrelated = np.array(engine.embed_text("قانون الأحوال المدنية والمحاكم"))

    assert len(v1) == 384
    # Check L2 unit norm
    assert np.isclose(np.linalg.norm(v1), 1.0, atol=1e-4)

    # Cosine similarity
    sim_related = np.dot(v1, v2)
    sim_unrelated = np.dot(v1, v_unrelated)

    # Related catering query should score higher than unrelated legal query
    assert sim_related > sim_unrelated


def test_qdrant_vector_store_in_memory():
    q_client = QdrantClient(location=":memory:")
    store = QdrantVectorStore(client=q_client, collection_name="test_tender_chunks")

    chunks = [
        DocumentChunk(
            chunk_id="chk_1",
            document_id="doc_food_1",
            page_number=1,
            chunk_type="text",
            content="وجبة الإفطار: جبن أبيض 65 جرام مع بيض وخبز بلدي",
            char_count=45,
            metadata={"item": "cheese"},
        ),
        DocumentChunk(
            chunk_id="chk_2",
            document_id="doc_food_1",
            page_number=2,
            chunk_type="text",
            content="تطبق غرامة تأخير 1% عن كل ساعة تأخير في تسليم الوجبة",
            char_count=52,
            metadata={"type": "penalty"},
        ),
        DocumentChunk(
            chunk_id="chk_3",
            document_id="doc_other_2",
            page_number=1,
            chunk_type="text",
            content="شروط التأمين الابتدائي وخطاب الضمان البنكي",
            char_count=40,
        ),
    ]

    indexed_count = store.index_chunks(chunks)
    assert indexed_count == 3
    assert store.count_document_chunks("doc_food_1") == 2

    # Query 1: Search for breakfast cheese
    results = store.search(query="كم جرام جبن في الفطار؟", document_id="doc_food_1", top_k=2)
    assert len(results) >= 1
    assert results[0].page_number == 1
    assert "جبن" in results[0].content
    assert results[0].score > 0.0

    # Query 2: Search for penalty
    results_penalty = store.search(query="غرامات التأخير والجزاءات", top_k=2)
    assert any("غرامة تأخير" in r.content for r in results_penalty)

    # Delete document
    deleted = store.delete_document("doc_food_1")
    assert deleted is True
    assert store.count_document_chunks("doc_food_1") == 0


def test_search_api_endpoint():
    response = client.post(
        "/api/v1/search",
        json={"query": "شروط الوجبات الغذائية", "top_k": 3},
    )
    assert response.status_code == 200
    data = response.json()
    assert "query" in data
    assert "results" in data
    assert data["query"] == "شروط الوجبات الغذائية"
