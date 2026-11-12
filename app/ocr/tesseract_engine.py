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
        Enhances image readability for OCR:
        1. Convert to grayscale.
        2. Enhance contrast.
        3. Sharpen edges.
        """
        # Convert to grayscale
        gray = image.convert("L")

        # Increase contrast to make faint Arabic text and stamps clear
        enhancer = ImageEnhance.Contrast(gray)
        contrast = enhancer.enhance(1.8)

        # Slight sharpening
        sharpened = contrast.filter(ImageFilter.SHARPEN)
        return sharpened

    def extract_from_image(self, image: Image.Image) -> Tuple[str, float]:
        """
        Runs OCR on a PIL Image and returns extracted text and mean confidence score (0.0 to 1.0).
        """
        if not self.is_available():
            return "", 0.0

        try:
            processed_img = self.preprocess_image(image)

            # PSM 1: Automatic page segmentation with OSD
            # PSM 3: Fully automatic page segmentation (default)
            # PSM 6: Assume a single uniform block of text
            custom_config = "--psm 3"

            # Use image_to_data to obtain token-level confidence scores
            data = pytesseract.image_to_data(
                processed_img,
                lang=self.languages,
                config=custom_config,
                output_type=pytesseract.Output.DICT,
            )

            text_tokens = []
            confidences = []

            for i in range(len(data["text"])):
                word = data["text"][i].strip()
                conf = float(data["conf"][i])

                # Tesseract marks confidence as -1 for empty/delimiter boxes
                if word and conf >= 0:
                    text_tokens.append(word)
                    confidences.append(conf)

            full_text = " ".join(text_tokens)
            mean_conf = (
                (sum(confidences) / len(confidences) / 100.0) if confidences else 0.0
            )

            return full_text, round(mean_conf, 4)

        except Exception as e:
            # Fallback on raw image_to_string if detailed data extraction fails
            try:
                raw_text = pytesseract.image_to_string(
                    image, lang=self.languages, config="--psm 3"
                )
                return raw_text.strip(), 0.5
            except Exception:
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
