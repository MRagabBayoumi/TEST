import json
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query, status
from app.core.config import settings
from app.document_processing.semantic_chunker import ArabicSemanticChunker
from app.schemas.chunk import SearchQueryRequest, SearchQueryResponse
from app.schemas.document import DocumentExtractionResult, PageContent
from app.vector_store.qdrant_service import get_vector_store

router = APIRouter(tags=["Semantic Search & Vector Indexing"])


@router.post(
    "/documents/{document_id}/index",
    summary="Index document chunks into Qdrant vector database",
    status_code=status.HTTP_200_OK,
)
def index_document(document_id: str):
    """
    Performs semantic chunking on an extracted document (keeping tables intact)
    and indexes all chunks into the local Qdrant vector database.
    """
    json_path = settings.PROCESSED_DIR / f"{document_id}.json"
    if not json_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"المستند المعالج غير موجود: {document_id}",
        )

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        pages = [PageContent(**p) for p in data.get("pages", [])]
        summary = data.get("summary", {})
        filename = summary.get("filename", "")

        chunker = ArabicSemanticChunker(target_chunk_size=500, chunk_overlap=80)
        chunks = chunker.chunk_document(pages, document_id=document_id, filename=filename)

        vector_store = get_vector_store()
        # Clean existing points for this doc before re-indexing
        vector_store.delete_document(document_id)
        indexed_count = vector_store.index_chunks(chunks)

        # Save chunks locally for inspection
        chunks_json_path = settings.PROCESSED_DIR / f"{document_id}_chunks.json"
        with open(chunks_json_path, "w", encoding="utf-8") as f:
            json.dump([c.model_dump(mode="json") for c in chunks], f, ensure_ascii=False, indent=2)

        return {
            "success": True,
            "document_id": document_id,
            "filename": filename,
            "total_chunks": len(chunks),
            "indexed_count": indexed_count,
            "message": "تم تقطيع وفهرسة المستند في قاعدة بيانات المتجهات Qdrant بنجاح",
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"فشل في فهرسة المستند: {str(e)}",
        )


@router.post(
    "/search",
    response_model=SearchQueryResponse,
    summary="Semantic vector search across indexed tender chunks and tables",
)
def semantic_search(request: SearchQueryRequest):
    """
    Searches indexed chunks using dense vector cosine similarity.
    Supports filtering by document_id and chunk_type ('text' or 'table').
    """
    vector_store = get_vector_store()
    results = vector_store.search(
        query=request.query,
        document_id=request.document_id,
        top_k=request.top_k,
        chunk_type=request.chunk_type,
    )

    return SearchQueryResponse(
        query=request.query,
        total_found=len(results),
        results=results,
    )
