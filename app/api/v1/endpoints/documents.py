import json
import uuid
from pathlib import Path
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from app.core.config import settings
from app.core.security import sanitize_filename, validate_pdf_content
from app.document_processing.extractor import PDFTextExtractor
from app.schemas.document import DocumentExtractionResult, PageContent

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
    saves the file safely in 1MB chunks to prevent memory spikes, extracts text page-by-page preserving
    page numbers, and stores structured JSON.
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

    try:
        with open(target_path, "wb") as out_file:
            while chunk := await file.read(1024 * 1024):  # 1MB buffer chunks
                total_bytes += len(chunk)
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

    # Run extraction pipeline
    try:
        extractor = PDFTextExtractor(
            pdf_path=target_path,
            document_id=document_id,
            original_filename=safe_filename,
        )
        result = extractor.extract_all(save_json=True)
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
