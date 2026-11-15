import io
import os
from pathlib import Path
from typing import Tuple
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import pymupdf
import pytesseract
from app.core.config import settings
from app.ocr.base import BaseOCREngine


class TesseractOCREngine(BaseOCREngine):
    """
    Tesseract OCR implementation supporting Arabic and multilingual documents.
    Features automated table grid line removal (inpainting) so table lines never
    bisect Arabic words, plus multi-PSM evaluation for maximum text completeness.
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

    def extract_from_image(
        self, image: Image.Image, psm_modes: list[int] | None = None
    ) -> Tuple[str, float]:
        """
        Runs OCR on a PIL Image and returns extracted text and mean confidence score (0.0 to 1.0).
        Evaluates candidate configurations and picks the one yielding the most complete text.
        """
        if not self.is_available():
            return "", 0.0

        modes = psm_modes or [6, 3]
        best_text = ""
        best_conf = 0.0

        candidates = [image, self.preprocess_image(image)]

        for psm in modes:
            for img in candidates:
                try:
                    text = pytesseract.image_to_string(
                        img,
                        lang=self.languages,
                        config=f"--psm {psm}",
                    ).strip()

                    if len(text) > len(best_text):
                        best_text = text
                        # Estimate confidence
                        try:
                            data = pytesseract.image_to_data(
                                img,
                                lang=self.languages,
                                config=f"--psm {psm}",
                                output_type=pytesseract.Output.DICT,
                            )
                            confs = [
                                float(c)
                                for w, c in zip(data["text"], data["conf"])
                                if w.strip() and float(c) >= 0
                            ]
                            best_conf = (
                                (sum(confs) / len(confs) / 100.0) if confs else 0.75
                            )
                        except Exception:
                            best_conf = 0.75

                except Exception:
                    continue

        return best_text, round(best_conf, 4)

    def extract_from_page(
        self, page: pymupdf.Page, dpi: int = 200
    ) -> Tuple[str, float]:
        """
        Renders a PyMuPDF Page to high-DPI image, performs morphological table grid
        line removal so lines don't bisect Arabic text, and runs OCR for maximum text completeness.
        """
        # Render high-resolution pixmap
        pix = page.get_pixmap(dpi=dpi)
        np_arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, 3
        )
        gray = cv2.cvtColor(np_arr, cv2.COLOR_RGB2GRAY)

        # Detect table lines
        thresh = cv2.adaptiveThreshold(
            ~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2
        )
        h_k = max(20, int(dpi * 0.18))
        v_k = max(20, int(dpi * 0.18))
        h_lines = cv2.morphologyEx(
            thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (h_k, 1))
        )
        v_lines = cv2.morphologyEx(
            thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_k))
        )
        lines_mask = cv2.dilate(
            h_lines + v_lines,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        )

        has_table_grid = np.sum(lines_mask > 0) > (pix.width * 2)

        if has_table_grid:
            # Inpaint / remove table lines so letters and digits are not bisected
            cleaned_gray = gray.copy()
            cleaned_gray[lines_mask > 0] = 255
            image = Image.fromarray(cleaned_gray)
            # Use PSM 6 (uniform block/tabular) followed by PSM 3
            return self.extract_from_image(image, psm_modes=[6, 3])

        # Standard prose page without heavy table grids
        image = Image.fromarray(gray)
        return self.extract_from_image(image, psm_modes=[3, 6])

