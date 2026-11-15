import hashlib
import json
import uuid
from pathlib import Path
import anyio
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from app.core.config import settings
from app.core.security import sanitize_filename, validate_pdf_content
from app.document_processing.extractor import PDFTextExtractor
from app.document_processing.semantic_chunker import ArabicSemanticChunker
from app.schemas.document import DocumentExtractionResult, PageContent
from app.vector_store.qdrant_service import get_vector_store

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post(
    "/upload",
    response_model=DocumentExtractionResult,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and extract text from a tender PDF",
)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Receives a PDF document via streaming, validates security constraints (size, header, extension),
    computes content hash for instant deduplication/caching, saves safely in 1MB chunks,
    extracts text concurrently in a worker thread (non-blocking), and indexes in Qdrant.
    """
    filename = file.filename or "tender.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="نوع الملف غير مدعوم، يجب رفع ملف بصيغة PDF فقط.",
        )

    # Generate unique document ID and clean filename
    document_id = f"doc_{uuid.uuid4().hex[:10]}"
    safe_filename = sanitize_filename(filename)
    target_filename = f"{document_id}_{safe_filename}"
    target_path = settings.UPLOAD_DIR / target_filename

    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    total_bytes = 0
    first_chunk = True
    sha256_hasher = hashlib.sha256()

    try:
        with open(target_path, "wb") as out_file:
            while chunk := await file.read(1024 * 1024):  # 1MB buffer chunks
                total_bytes += len(chunk)
                sha256_hasher.update(chunk)
                if total_bytes > max_bytes:
                    out_file.close()
                    target_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"حجم الملف يتجاوز الحد الأقصى المسموح ({settings.MAX_UPLOAD_SIZE_MB} ميجابايت).",
                    )
                if first_chunk:
                    if not chunk.startswith(b"%PDF-"):
                        out_file.close()
                        target_path.unlink(missing_ok=True)
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="الملف ليس ملف PDF صالحاً (توقيع الهيدر غير مطابق).",
                        )
                    first_chunk = False
                out_file.write(chunk)

        if total_bytes == 0:
            target_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="الملف فارغ (0 بايت).",
            )
    except HTTPException:
        raise
    except Exception as e:
        target_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"حدث خطأ أثناء حفظ الملف: {str(e)}",
        )

    file_digest = sha256_hasher.hexdigest()
    hash_index_path = settings.PROCESSED_DIR / "hash_index.json"

    # Fast Cache Check: If identical file was already processed, return immediately
    if hash_index_path.exists():
        try:
            with open(hash_index_path, "r", encoding="utf-8") as f:
                hash_cache = json.load(f)
            if file_digest in hash_cache:
                cached_id = hash_cache[file_digest]
                cached_json = settings.PROCESSED_DIR / f"{cached_id}.json"
                if cached_json.exists():
                    target_path.unlink(missing_ok=True)  # Clean redundant upload file
                    with open(cached_json, "r", encoding="utf-8") as f:
                        cached_data = json.load(f)
                    return DocumentExtractionResult(
                        summary=cached_data["summary"],
                        pages=cached_data["pages"],
                        processed_json_path=str(cached_json),
                    )
        except Exception:
            pass

    # Run extraction pipeline asynchronously in threadpool to keep event loop responsive
    try:
        extractor = PDFTextExtractor(
            pdf_path=target_path,
            document_id=document_id,
            original_filename=safe_filename,
        )
        result = await anyio.to_thread.run_sync(extractor.extract_all, True)

        # Store content hash index for future instant reuse
        try:
            cache_map = {}
            if hash_index_path.exists():
                with open(hash_index_path, "r", encoding="utf-8") as f:
                    cache_map = json.load(f)
            cache_map[file_digest] = document_id
            with open(hash_index_path, "w", encoding="utf-8") as f:
                json.dump(cache_map, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        # Auto-index into Qdrant vector database (Stage 4)
        try:
            chunker = ArabicSemanticChunker(target_chunk_size=500, chunk_overlap=80)
            chunks = chunker.chunk_document(
                result.pages, document_id=document_id, filename=safe_filename
            )
            vector_store = get_vector_store()
            vector_store.delete_document(document_id)
            vector_store.index_chunks(chunks)
        except Exception:
            pass

        return result
    except Exception as e:
        # If extraction crashes, ensure we raise informative error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"حدث خطأ أثناء معالجة المستند: {str(e)}",
        )


@router.get(
    "/{document_id}",
    response_model=DocumentExtractionResult,
    summary="Get previously extracted document data",
)
async def get_document(document_id: str):
    """Retrieve the cached/processed JSON result of a previously uploaded document."""
    json_path = settings.PROCESSED_DIR / f"{document_id}.json"
    if not json_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"المستند ذو المعرف ({document_id}) غير موجود.",
        )

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return DocumentExtractionResult(
        summary=data["summary"],
        pages=data["pages"],
        processed_json_path=str(json_path),
    )


@router.get(
    "/{document_id}/pages/{page_number}",
    response_model=PageContent,
    summary="Get extracted text for a specific page",
)
async def get_document_page(document_id: str, page_number: int):
    """Retrieve text and metadata of a single page by its 1-indexed page number."""
    json_path = settings.PROCESSED_DIR / f"{document_id}.json"
    if not json_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"المستند ذو المعرف ({document_id}) غير موجود.",
        )

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pages = data.get("pages", [])
    for page in pages:
        if page.get("page_number") == page_number:
            return PageContent(**page)

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"الصفحة رقم ({page_number}) غير موجودة في هذا المستند.",
    )
