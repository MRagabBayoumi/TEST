import io
import os
from pathlib import Path
from typing import Tuple
from PIL import Image, ImageEnhance, ImageFilter
import pymupdf
import pytesseract
from app.core.config import settings
from app.ocr.base import BaseOCREngine


class TesseractOCREngine(BaseOCREngine):
    """
    Tesseract OCR implementation supporting Arabic and multilingual documents.
    Handles image preprocessing (contrast enhancement, binarization) and confidence scoring.
    """

    def __init__(
        self,
        tesseract_cmd: str | None = None,
        tessdata_dir: Path | None = None,
        languages: str | None = None,
    ):
        self.tesseract_cmd = tesseract_cmd or settings.TESSERACT_CMD
        self.tessdata_dir = tessdata_dir or settings.TESSDATA_DIR
        self.languages = languages or settings.OCR_LANGUAGES

        # Configure pytesseract binary and environment
        if Path(self.tesseract_cmd).exists():
            pytesseract.pytesseract.tesseract_cmd = str(self.tesseract_cmd)

        if self.tessdata_dir.exists():
            os.environ["TESSDATA_PREFIX"] = str(self.tessdata_dir.resolve())

    def is_available(self) -> bool:
        """Check if tesseract binary and Arabic model exist."""
        binary_exists = Path(self.tesseract_cmd).exists()
        arabic_model_exists = (self.tessdata_dir / "ara.traineddata").exists()
        return binary_exists and arabic_model_exists

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Enhances image readability for OCR with mild contrast optimization
        without destroying faint Arabic typography.
        """
        # Convert to grayscale
        gray = image.convert("L")

        # Mild contrast adjustment suitable for scanned documents
        enhancer = ImageEnhance.Contrast(gray)
        adjusted = enhancer.enhance(1.2)
        return adjusted

    def extract_from_image(self, image: Image.Image) -> Tuple[str, float]:
        """
        Runs OCR on a PIL Image and returns extracted text and mean confidence score (0.0 to 1.0).
        Uses a robust multi-pass strategy (raw image -> preprocessed image) with fallback to image_to_string.
        """
        if not self.is_available():
            return "", 0.0

        candidates = [image, self.preprocess_image(image)]

        for img in candidates:
            try:
                # First attempt: image_to_string with PSM 3 (most reliable for Arabic text layout)
                text = pytesseract.image_to_string(
                    img,
                    lang=self.languages,
                    config="--psm 3",
                ).strip()

                if text:
                    # Calculate approximate confidence via image_to_data
                    try:
                        data = pytesseract.image_to_data(
                            img,
                            lang=self.languages,
                            config="--psm 3",
                            output_type=pytesseract.Output.DICT,
                        )
                        confs = [
                            float(c)
                            for w, c in zip(data["text"], data["conf"])
                            if w.strip() and float(c) >= 0
                        ]
                        mean_conf = (
                            (sum(confs) / len(confs) / 100.0) if confs else 0.75
                        )
                    except Exception:
                        mean_conf = 0.75

                    return text, round(mean_conf, 4)

            except Exception:
                continue

        return "", 0.0

    def extract_from_page(self, page: pymupdf.Page, dpi: int = 200) -> Tuple[str, float]:
        """
        Renders a PyMuPDF Page to high-DPI raster image and performs OCR.
        """
        # Render high-resolution pixmap
        pix = page.get_pixmap(dpi=dpi)
        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes))

        return self.extract_from_image(image)
