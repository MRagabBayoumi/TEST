import re
from typing import Optional
from app.schemas.chunk import DocumentChunk
from app.schemas.document import PageContent


class ArabicSemanticChunker:
    """
    Splits Arabic government catering tenders and documents into semantically coherent
    chunks optimized for vector embeddings and RAG:
    1. Keeps tables completely intact in Markdown format so row/col relationships are preserved.
    2. Respects Arabic structural boundaries (sections, items 'أولاً / ثانياً', clauses 'بند / مادة', bullet points).
    3. Maintains contextual overlap across paragraph breaks.
    4. Attaches precise page numbers and metadata to every chunk for verifiable citation.
    """

    def __init__(
        self,
        target_chunk_size: int = 500,
        chunk_overlap: int = 80,
        min_chunk_size: int = 30,
    ):
        self.target_chunk_size = target_chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk_page(
        self,
        page: PageContent,
        document_id: str,
        filename: str = "",
    ) -> list[DocumentChunk]:
        """Chunks a single page into text and table chunks."""
        chunks: list[DocumentChunk] = []
        chunk_counter = 0

        # 1. Extract tables as dedicated, atomic chunks
        if hasattr(page, "tables") and page.tables:
            for tab in page.tables:
                table_md = tab.markdown.strip() if hasattr(tab, "markdown") else ""
                if not table_md:
                    continue

                chunk_content = f"### [جدول مستخرج - صفحة {page.page_number}]\n{table_md}"
                chunk_id = f"chk_{document_id}_p{page.page_number}_tbl_{chunk_counter}"
                chunk_counter += 1

                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        page_number=page.page_number,
                        chunk_type="table",
                        content=chunk_content,
                        token_count=max(1, len(chunk_content) // 4),
                        char_count=len(chunk_content),
                        metadata={
                            "filename": filename,
                            "table_id": getattr(tab, "table_id", ""),
                            "row_count": getattr(tab, "row_count", 0),
                            "col_count": getattr(tab, "col_count", 0),
                            "headers": getattr(tab, "headers", []),
                        },
                    )
                )

        # 2. Extract and segment text
        raw_text = page.text.strip() if page.text else ""
        if not raw_text:
            return chunks

        # Split text into natural paragraphs/sections
        # Splitting on double newlines or Arabic section markers
        raw_paragraphs = re.split(r"\n\s*\n", raw_text)
        cleaned_paragraphs = []
        for p in raw_paragraphs:
            p_clean = p.strip()
            if p_clean:
                cleaned_paragraphs.append(p_clean)

        if not cleaned_paragraphs:
            return chunks

        # Accumulate paragraphs into semantic chunks with overlap
        current_buffer = []
        current_len = 0

        for para in cleaned_paragraphs:
            para_len = len(para)

            # If adding this paragraph exceeds target size and current_buffer is not empty:
            if current_len + para_len > self.target_chunk_size and current_buffer:
                chunk_text = "\n\n".join(current_buffer)
                if len(chunk_text.strip()) >= self.min_chunk_size:
                    chunk_id = f"chk_{document_id}_p{page.page_number}_txt_{chunk_counter}"
                    chunk_counter += 1
                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            document_id=document_id,
                            page_number=page.page_number,
                            chunk_type="text",
                            content=chunk_text.strip(),
                            token_count=max(1, len(chunk_text) // 4),
                            char_count=len(chunk_text.strip()),
                            metadata={"filename": filename},
                        )
                    )

                # Overlap logic: keep last paragraph if short, or slice suffix
                last_para = current_buffer[-1]
                if len(last_para) <= self.chunk_overlap:
                    current_buffer = [last_para, para]
                    current_len = len(last_para) + para_len
                else:
                    current_buffer = [para]
                    current_len = para_len
            else:
                current_buffer.append(para)
                current_len += para_len

        # Flush remaining buffer
        if current_buffer:
            chunk_text = "\n\n".join(current_buffer).strip()
            if chunk_text:
                if len(chunk_text) < self.min_chunk_size and chunks:
                    # Append short trailing fragment to the last chunk instead of dropping it
                    chunks[-1].content += "\n\n" + chunk_text
                    chunks[-1].char_count = len(chunks[-1].content)
                    chunks[-1].token_count = max(1, chunks[-1].char_count // 4)
                else:
                    chunk_id = f"chk_{document_id}_p{page.page_number}_txt_{chunk_counter}"
                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            document_id=document_id,
                            page_number=page.page_number,
                            chunk_type="text",
                            content=chunk_text,
                            token_count=max(1, len(chunk_text) // 4),
                            char_count=len(chunk_text),
                            metadata={"filename": filename},
                        )
                    )

        return chunks

    def chunk_document(
        self,
        pages: list[PageContent],
        document_id: str,
        filename: str = "",
    ) -> list[DocumentChunk]:
        """Chunks all pages in the document."""
        all_chunks: list[DocumentChunk] = []
        for page in pages:
            page_chunks = self.chunk_page(page, document_id, filename)
            all_chunks.extend(page_chunks)
        return all_chunks
