from typing import Any, Optional
from pydantic import BaseModel, Field


class CitationItem(BaseModel):
    """Represents a verifiable source citation referencing an exact page and chunk."""

    page_number: int = Field(..., description="1-indexed source page number in the original tender")
    chunk_id: str = Field(..., description="Unique ID of the cited chunk")
    chunk_type: str = Field(..., description="'text' or 'table'")
    snippet: str = Field(..., description="Excerpt from the source document supporting the answer")
    score: float = Field(..., description="Semantic relevance score (0.0 to 1.0)")


class RAGQuestionRequest(BaseModel):
    """Request payload for RAG question answering."""

    question: str = Field(
        ...,
        min_length=2,
        max_length=500,
        description="User's natural language question about the tender specifications",
    )
    document_id: Optional[str] = Field(
        default=None,
        description="Target document ID to answer from; searches all if omitted",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=15,
        description="Number of context chunks to retrieve for synthesis",
    )


class RAGAnswerResponse(BaseModel):
    """Structured response containing synthesized answer and verifiable page citations."""

    question: str
    answer: str
    citations: list[CitationItem]
    retrieved_chunks_count: int
