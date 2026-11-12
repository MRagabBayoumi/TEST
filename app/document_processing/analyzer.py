from pathlib import Path
import pymupdf  # Modern PyMuPDF



class PDFAnalyzer:
    """Analyzes PDF files for page structure, digital text availability, and image content."""

    def __init__(self, pdf_path: Path):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found at: {self.pdf_path}")

    def inspect_document(self) -> dict:
        """
        Inspects the entire PDF and gathers page-by-page structural metrics.
        Returns a dictionary containing summary counts and page characteristics.
        """
        doc = pymupdf.open(self.pdf_path)
        total_pages = len(doc)
        digital_pages_count = 0
        scanned_pages_count = 0
        page_analysis = []

        for page_index in range(total_pages):
            page = doc.load_page(page_index)
            page_num = page_index + 1

            # Extract raw text layer
            raw_text = page.get_text("text").strip()
            char_count = len(raw_text)
            word_count = len(raw_text.split())

            # Detect embedded images
            image_list = page.get_images(full=True)
            image_count = len(image_list)

            # Heuristic for scanned pages:
            # If the page has very few characters (< 40) and has images, or has 0 characters
            is_scanned = (char_count < 40 and image_count > 0) or (char_count == 0)

            if is_scanned:
                scanned_pages_count += 1
            else:
                digital_pages_count += 1

            rect = page.rect
            page_analysis.append(
                {
                    "page_number": page_num,
                    "char_count": char_count,
                    "word_count": word_count,
                    "has_text": char_count > 0,
                    "is_scanned_candidate": is_scanned,
                    "image_count": image_count,
                    "width": float(rect.width),
                    "height": float(rect.height),
                }
            )

        doc.close()

        return {
            "total_pages": total_pages,
            "digital_pages_count": digital_pages_count,
            "scanned_pages_count": scanned_pages_count,
            "pages": page_analysis,
        }
