import uuid
from typing import Optional
from qdrant_client import QdrantClient, models
from app.core.config import settings
from app.schemas.chunk import DocumentChunk, SearchResultItem
from app.vector_store.embeddings import BaseEmbeddingEngine, get_embedding_engine


class QdrantVectorStore:
    """
    Manages vector storage, indexing, and semantic similarity search using Qdrant.
    Operates in local file-embedded storage mode (storage/qdrant_db) or in-memory.
    """

    COLLECTION_NAME = "tender_chunks"

    def __init__(
        self,
        client: Optional[QdrantClient] = None,
        embedding_engine: Optional[BaseEmbeddingEngine] = None,
        collection_name: str = COLLECTION_NAME,
    ):
        self.embedding_engine = embedding_engine or get_embedding_engine()
        self.collection_name = collection_name

        if client is not None:
            self.client = client
        else:
            # File-based local persistent Qdrant
            db_path = str(settings.QDRANT_DIR.resolve())
            self.client = QdrantClient(path=db_path)

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Creates the Qdrant collection if it does not already exist."""
        collections_resp = self.client.get_collections()
        existing_names = [c.name for c in collections_resp.collections]

        if self.collection_name not in existing_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.embedding_engine.dimension,
                    distance=models.Distance.COSINE,
                ),
            )
            # Create payload index for server Qdrant (skip for local/memory client to prevent UserWarning)
            try:
                client_backend = getattr(self.client, "_client", None)
                if type(client_backend).__name__ != "QdrantLocal":
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="document_id",
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="chunk_type",
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
            except Exception:
                pass

    def index_chunks(self, chunks: list[DocumentChunk]) -> int:
        """
        Generates dense embeddings for chunks and upserts them into Qdrant.
        Returns total number of points successfully indexed.
        """
        if not chunks:
            return 0

        points: list[models.PointStruct] = []
        texts_to_embed = [c.content for c in chunks]
        embeddings = self.embedding_engine.embed_batch(texts_to_embed)

        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb
            point_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))

            payload = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "page_number": chunk.page_number,
                "chunk_type": chunk.chunk_type,
                "content": chunk.content,
                "token_count": chunk.token_count,
                "char_count": chunk.char_count,
                "metadata": chunk.metadata,
                "created_at": chunk.created_at.isoformat(),
            }

            points.append(
                models.PointStruct(
                    id=point_uuid,
                    vector=emb,
                    payload=payload,
                )
            )

        # Batch upsert
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )

        return len(points)

    def search(
        self,
        query: str,
        document_id: Optional[str] = None,
        top_k: int = 5,
        chunk_type: Optional[str] = None,
    ) -> list[SearchResultItem]:
        """
        Performs semantic cosine similarity search over indexed chunks with optional filtering.
        """
        if not query or not query.strip():
            return []

        query_vector = self.embedding_engine.embed_text(query)

        filter_conditions = []
        if document_id:
            filter_conditions.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id),
                )
            )
        if chunk_type:
            filter_conditions.append(
                models.FieldCondition(
                    key="chunk_type",
                    match=models.MatchValue(value=chunk_type),
                )
            )

        search_filter = (
            models.Filter(must=filter_conditions) if filter_conditions else None
        )

        query_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=search_filter,
            limit=top_k,
            with_payload=True,
        )

        results: list[SearchResultItem] = []
        for point in query_result.points:
            payload = point.payload or {}
            # Normalize Cosine similarity score (0.0 to 1.0)
            score = max(0.0, min(1.0, float(point.score)))

            results.append(
                SearchResultItem(
                    chunk_id=str(payload.get("chunk_id", point.id)),
                    document_id=str(payload.get("document_id", "")),
                    page_number=int(payload.get("page_number", 1)),
                    chunk_type=str(payload.get("chunk_type", "text")),
                    content=str(payload.get("content", "")),
                    score=round(score, 4),
                    metadata=dict(payload.get("metadata", {})),
                )
            )

        return results

    def count_document_chunks(self, document_id: str) -> int:
        """Returns total chunks stored for a specific document."""
        count_res = self.client.count(
            collection_name=self.collection_name,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ]
            ),
        )
        return count_res.count

    def delete_document(self, document_id: str) -> bool:
        """Deletes all chunks belonging to a document from the vector store."""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                ),
            )
            return True
        except Exception:
            return False


_qdrant_instance: QdrantVectorStore | None = None


def get_vector_store() -> QdrantVectorStore:
    """Returns singleton instance of the Qdrant vector store."""
    global _qdrant_instance
    if _qdrant_instance is None:
        _qdrant_instance = QdrantVectorStore()
    return _qdrant_instance
