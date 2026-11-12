from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field


class PageContent(BaseModel):
    """Represents the extracted text and metadata of a single PDF page."""

    page_number: int = Field(..., description="1-indexed page number from the original document")
    char_count: int = Field(..., description="Number of text characters found on the page")
    word_count: int = Field(..., description="Number of words found on the page")
    has_text: bool = Field(..., description="Whether the page contains text (digital or OCR)")
    is_scanned_candidate: bool = Field(
        ...,
        description="True if page was detected as a scanned or image-based page",
    )
    image_count: int = Field(default=0, description="Number of embedded raster images on this page")
    extraction_method: str = Field(
        default="digital",
        description="Method used for extraction: 'digital' or 'ocr'",
    )
    ocr_confidence: Optional[float] = Field(
        default=None,
        description="Average confidence score from OCR engine (0.0 to 1.0)",
    )
    width: float = Field(..., description="Page width in points")
    height: float = Field(..., description="Page height in points")
    text: str = Field(..., description="Extracted and cleaned text from the page")


class DocumentSummary(BaseModel):
    """Overall summary of the processed PDF document."""

    document_id: str = Field(..., description="Unique identifier for the document")
    filename: str = Field(..., description="Sanitized original file name")
    file_size_bytes: int = Field(..., description="Total size in bytes")
    total_pages: int = Field(..., description="Total page count")
    digital_pages_count: int = Field(
        ..., description="Number of pages containing digital text layers"
    )
    scanned_pages_count: int = Field(
        ..., description="Number of pages detected as scanned or image-only"
    )
    ocr_pages_count: int = Field(
        default=0, description="Number of pages processed and recovered via OCR"
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DocumentExtractionResult(BaseModel):
    """Complete extraction result including all pages and summary."""

    summary: DocumentSummary
    pages: list[PageContent]
    processed_json_path: Optional[str] = Field(
        None, description="Path where the structured JSON was saved"
    )
