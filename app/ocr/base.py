from abc import ABC, abstractmethod
from typing import Tuple
from PIL import Image
import pymupdf


class BaseOCREngine(ABC):
    """
    Abstract interface for OCR engines.
    Ensures that any OCR provider (Tesseract, PaddleOCR, EasyOCR, or Cloud OCR)
    can be swapped without altering the rest of the application.
    """

    @abstractmethod
    def is_available(self) -> bool:
        """Returns True if the engine binaries and language models are ready."""
        pass

    @abstractmethod
    def extract_from_image(self, image: Image.Image) -> Tuple[str, float]:
        """
        Extract text and average confidence score from a PIL Image.
        Returns:
            Tuple[str, float]: (extracted_text, confidence_score between 0.0 and 1.0)
        """
        pass

    @abstractmethod
    def extract_from_page(self, page: pymupdf.Page, dpi: int = 200) -> Tuple[str, float]:
        """
        Renders a PDF page to a high-DPI pixmap and runs OCR.
        Returns:
            Tuple[str, float]: (extracted_text, confidence_score between 0.0 and 1.0)
        """
        pass
