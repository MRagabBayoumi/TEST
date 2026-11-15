import os
import re
from typing import Optional
import httpx
from app.core.config import settings
from app.schemas.rag import CitationItem, RAGAnswerResponse, RAGQuestionRequest
from app.vector_store.qdrant_service import QdrantVectorStore, get_vector_store


class TenderRAGService:
    """
    RAG (Retrieval-Augmented Generation) QA Engine for Tender AI Lab.
    1. Retrieves top semantic text chunks and table grids from Qdrant.
    2. Builds verified page citations referencing exact page numbers and tables.
    3. Synthesizes an accurate Arabic answer using either:
       - Neural LLM (Gemini Flash free tier if GEMINI_API_KEY is configured in .env).
       - Local Extractive Summarizer (100% offline and free, zero config).
    """

    def __init__(
        self,
        vector_store: Optional[QdrantVectorStore] = None,
        gemini_api_key: Optional[str] = None,
    ):
        self.vector_store = vector_store or get_vector_store()
        self.api_key = gemini_api_key or os.getenv("GEMINI_API_KEY", "")

    def ask(self, request: RAGQuestionRequest) -> RAGAnswerResponse:
        """Processes a natural language question and generates a cited answer."""
        retrieved = self.vector_store.search(
            query=request.question,
            document_id=request.document_id,
            top_k=request.top_k,
        )

        if not retrieved:
            return RAGAnswerResponse(
                question=request.question,
                answer="لم يتم العثور على نصوص أو جداول مطابقة في كراسة الشروط للإجابة على هذا السؤال.",
                citations=[],
                retrieved_chunks_count=0,
            )

        # Build verifiable citations
        citations: list[CitationItem] = []
        context_blocks: list[str] = []

        for item in retrieved:
            # Clean snippet for display
            snippet = item.content.strip()
            if len(snippet) > 280:
                snippet = snippet[:280] + "..."

            citations.append(
                CitationItem(
                    page_number=item.page_number,
                    chunk_id=item.chunk_id,
                    chunk_type=item.chunk_type,
                    snippet=snippet,
                    score=item.score,
                )
            )

            context_blocks.append(
                f"[صفحة {item.page_number} - نوع المحتوى: {item.chunk_type}]\n{item.content}"
            )

        full_context = "\n\n---\n\n".join(context_blocks)

        # Generate answer: try Neural LLM if key exists, else use Local Extractive Synthesizer
        answer = None
        if self.api_key:
            answer = self._generate_with_gemini(request.question, full_context)

        if not answer:
            answer = self._generate_local_extractive_answer(request.question, retrieved)

        return RAGAnswerResponse(
            question=request.question,
            answer=answer,
            citations=citations,
            retrieved_chunks_count=len(retrieved),
        )

    def _generate_with_gemini(self, question: str, context: str) -> Optional[str]:
        """Synthesizes answer using Google Gemini Flash Free Tier."""
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.api_key}"
            system_instruction = (
                "أنت خبير معتمد في كراسات الشروط والمناقصات الحكومية الخاصة بتوريد الأغذية والكيترينج لمستشفيات كايان.\n"
                "أجب عن سؤال المستخدم باللغة العربية بأسلوب مهني ومحدد وموثق استناداً فقط إلى السياق المرفق.\n"
                "إذا كانت الإجابة في جدول أو بنود، اذكر الأصناف، الجرامات، نسب الغرامات، وأرقام الصفحات بدقة."
            )

            prompt = f"{system_instruction}\n\nالسياق المستخرج من كراسة الشروط:\n{context}\n\nالسؤال:\n{question}"

            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.2, "maxOutputTokens": 800},
            }

            with httpx.Client(timeout=15.0) as client:
                res = client.post(url, json=payload)
                if res.status_code == 200:
                    data = res.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            return parts[0].get("text", "").strip()
        except Exception:
            pass

        return None

    def _generate_local_extractive_answer(self, question: str, retrieved: list) -> str:
        """
        Local fallback summarizer (100% offline & free):
        Extracts relevant sentences, table rows, and numerical specs with page citations.
        """
        lines = []
        pages_cited = sorted(list(set(r.page_number for r in retrieved)))
        pages_str = "، ".join([f"صفحة {p}" for p in pages_cited])

        lines.append(f"استناداً إلى كراسة الشروط المرفوعة ({pages_str})، إليك أهم البيانات المستخرجة:")
        lines.append("")

        added_snippets = set()
        for idx, item in enumerate(retrieved[:3]):
            content = item.content.strip()
            # Clean markdown table prefix if present
            if content.startswith("###"):
                content = content.split("\n", 1)[-1].strip()

            # Take key lines
            sub_lines = [l.strip() for l in content.split("\n") if l.strip()]
            for sl in sub_lines:
                if len(sl) > 15 and sl not in added_snippets and not sl.startswith("| ---"):
                    added_snippets.add(sl)
                    if sl.startswith("|"):
                        lines.append(f"• **جدول مقننات (صفحة {item.page_number})**: {sl}")
                    else:
                        lines.append(f"• **(صفحة {item.page_number})**: {sl}")

                if len(lines) >= 8:
                    break
            if len(lines) >= 8:
                break

        return "\n".join(lines)


_rag_service_instance: TenderRAGService | None = None


def get_rag_service() -> TenderRAGService:
    """Singleton getter for TenderRAGService."""
    global _rag_service_instance
    if _rag_service_instance is None:
        _rag_service_instance = TenderRAGService()
    return _rag_service_instance
