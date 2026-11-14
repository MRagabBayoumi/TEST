from datetime import datetime, timezone
from typing import Any, Optional
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """Represents a semantically bounded chunk of text or table ready for vector indexing."""

    chunk_id: str = Field(..., description="Unique chunk identifier (e.g. chk_doc123_p2_0)")
    document_id: str = Field(..., description="Identifier of parent document")
    page_number: int = Field(..., description="1-indexed source page number")
    chunk_type: str = Field(
        default="text",
        description="Type of chunk: 'text', 'table', or 'heading'",
    )
    content: str = Field(..., description="Text content or Markdown representation of the chunk")
    token_count: int = Field(default=0, description="Estimated token count")
    char_count: int = Field(..., description="Character count of content")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Rich metadata (table_id, original_filename, headers, section_title, etc.)",
    )
    embedding: Optional[list[float]] = Field(
        default=None, description="Dense vector embedding of the chunk content"
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SearchResultItem(BaseModel):
    """Represents a single ranked search result from Qdrant vector store."""

    chunk_id: str
    document_id: str
    page_number: int
    chunk_type: str
    content: str
    score: float = Field(..., description="Cosine similarity score (0.0 to 1.0)")
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchQueryRequest(BaseModel):
    """Payload for semantic search query."""

    query: str = Field(..., min_length=2, max_length=500, description="Natural language search question")
    document_id: Optional[str] = Field(
        default=None, description="Optional document filter; searches all if omitted"
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Number of relevant chunks to retrieve")
    chunk_type: Optional[str] = Field(
        default=None, description="Optional filter by chunk type: 'text' or 'table'"
    )


class SearchQueryResponse(BaseModel):
    """Response containing ranked semantic search results."""

    query: str
    total_found: int
    results: list[SearchResultItem]
