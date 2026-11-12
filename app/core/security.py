import re
import uuid
from pathlib import Path
from app.core.config import settings

# PDF magic bytes in binary header
PDF_MAGIC_BYTES = b"%PDF-"


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent directory traversal and special character exploits,
    while preserving Arabic and alphanumeric characters.
    """
    # Remove directory paths
    clean_name = Path(filename).name
    # Keep Arabic characters, English letters, digits, dots, dashes, and underscores
    clean_name = re.sub(r"[^\w\s\u0600-\u06FF\.\-_]", "", clean_name)
    # Replace whitespace with underscore
    clean_name = re.sub(r"\s+", "_", clean_name.strip())
    # Ensure it ends with .pdf
    if not clean_name.lower().endswith(".pdf"):
        clean_name += ".pdf"
    return clean_name or f"document_{uuid.uuid4().hex[:8]}.pdf"


def validate_pdf_content(file_bytes: bytes, filename: str) -> tuple[bool, str]:
    """
    Validate that:
    1. The file has a .pdf extension.
    2. The file is not empty.
    3. The file does not exceed MAX_UPLOAD_SIZE_MB.
    4. The file begins with valid PDF magic bytes (%PDF-).
    """
    # Check extension
    if not filename.lower().endswith(".pdf"):
        return False, "نوع الملف غير مدعوم، يجب رفع ملف بصيغة PDF فقط."

    # Check empty file
    file_size_bytes = len(file_bytes)
    if file_size_bytes == 0:
        return False, "الملف فارغ (0 بايت)."

    # Check maximum allowed size
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if file_size_bytes > max_bytes:
        return (
            False,
            f"حجم الملف ({file_size_bytes / (1024 * 1024):.2f} ميجابايت) يتجاوز الحد الأقصى المسموح ({settings.MAX_UPLOAD_SIZE_MB} ميجابايت).",
        )

    # Check PDF magic bytes (header validation)
    if not file_bytes.startswith(PDF_MAGIC_BYTES):
        return False, "الملف ليس ملف PDF صالحاً (توقيع الهيدر غير مطابق)."

    return True, "الملف صالح."
