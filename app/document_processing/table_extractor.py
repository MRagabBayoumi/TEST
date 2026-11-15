import os
from pathlib import Path
import re
from typing import Optional
import cv2
import numpy as np
from PIL import Image
import pymupdf
import pytesseract
from app.core.config import settings
from app.document_processing.cleaner import clean_arabic_text
from app.schemas.document import ExtractedTable


class TableExtractor:
    """
    Detects and extracts structured tables from both:
    1. Digital PDF vector pages (using PyMuPDF geometric table extraction).
    2. Scanned / image-based PDF pages (using OpenCV morphological line projection
       and spatial coordinate OCR cell reconstruction).
    Formats tabular data into rows, Markdown, and JSON-ready dicts for ERP ingestion.
    """

    @classmethod
    def extract_tables_from_page(
        cls,
        page: pymupdf.Page,
        page_number: int,
    ) -> list[ExtractedTable]:
        """
        Extracts structured tables from the page, checking vector lines first
        and automatically falling back to computer-vision scanned table extraction.
        """
        extracted_tables: list[ExtractedTable] = []

        # 1. Attempt vector-based table extraction (digital pages)
        try:
            tabs = page.find_tables(use_layout=False)
            if tabs and tabs.tables:
                for idx, tab in enumerate(tabs.tables):
                    table_id = f"tbl_p{page_number}_{idx + 1}"
                    raw_data = tab.extract()

                    if not raw_data or len(raw_data) == 0:
                        continue

                    cleaned_grid: list[list[str]] = []
                    for raw_row in raw_data:
                        cleaned_row = []
                        for cell in raw_row:
                            cell_str = str(cell).strip() if cell is not None else ""
                            if cell_str.lower() == "none":
                                cell_str = ""
                            cleaned_cell = clean_arabic_text(cell_str)
                            cleaned_row.append(cleaned_cell)
                        cleaned_grid.append(cleaned_row)

                    if not cleaned_grid or len(cleaned_grid) == 0:
                        continue

                    headers = [h if h else f"عمود_{i+1}" for i, h in enumerate(cleaned_grid[0])]
                    data_rows_grid = cleaned_grid[1:] if len(cleaned_grid) > 1 else []

                    dict_rows: list[dict[str, str]] = []
                    for row in data_rows_grid:
                        row_dict = {}
                        for col_idx, col_name in enumerate(headers):
                            val = row[col_idx] if col_idx < len(row) else ""
                            row_dict[col_name] = val
                        dict_rows.append(row_dict)

                    markdown_lines = [
                        "| " + " | ".join(headers) + " |",
                        "| " + " | ".join(["---"] * len(headers)) + " |",
                    ]
                    for r in data_rows_grid:
                        markdown_lines.append("| " + " | ".join(r) + " |")

                    bbox = list(tab.bbox) if hasattr(tab, "bbox") and tab.bbox else None

                    extracted_tables.append(
                        ExtractedTable(
                            table_id=table_id,
                            page_number=page_number,
                            row_count=len(cleaned_grid),
                            col_count=len(headers),
                            headers=headers,
                            rows=data_rows_grid,
                            bbox=bbox,
                            markdown="\n".join(markdown_lines),
                            data_rows=dict_rows,
                        )
                    )
        except Exception:
            pass

        # If vector tables found, return them
        if extracted_tables:
            return extracted_tables

        # 2. Fallback: Computer-Vision Scanned Table Detection
        scanned_tables = cls.extract_tables_from_scanned_page(page, page_number)
        return scanned_tables

    @classmethod
    def extract_tables_from_scanned_page(
        cls,
        page: pymupdf.Page,
        page_number: int,
        dpi: int = 200,
    ) -> list[ExtractedTable]:
        """
        Detects rasterized/scanned table grids on image pages using OpenCV morphological
        operations and groups OCR words by spatial coordinates into structured rows and columns.
        """
        results: list[ExtractedTable] = []

        # Configure Tesseract binary and environment
        if Path(settings.TESSERACT_CMD).exists():
            pytesseract.pytesseract.tesseract_cmd = str(settings.TESSERACT_CMD)
        if settings.TESSDATA_DIR.exists():
            os.environ["TESSDATA_PREFIX"] = str(settings.TESSDATA_DIR.resolve())

        try:
            pix = page.get_pixmap(dpi=dpi)
            np_arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            gray = cv2.cvtColor(np_arr, cv2.COLOR_RGB2GRAY)

            # Adaptive thresholding
            thresh = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

            h_k = max(20, int(dpi * 0.18))
            v_k = max(20, int(dpi * 0.18))
            h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (h_k, 1)))
            v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_k)))

            # Combine lines to find closed table bounding boxes
            table_mask = cv2.dilate(h_lines + v_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
            table_contours, _ = cv2.findContours(
                cv2.dilate(h_lines + v_lines, np.ones((5, 5), np.uint8)),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            # Filter for rectangular tables with significant size
            min_w = int(pix.width * 0.35)
            min_h = int(pix.height * 0.08)
            candidate_tables = [
                cv2.boundingRect(c)
                for c in table_contours
                if cv2.boundingRect(c)[2] >= min_w and cv2.boundingRect(c)[3] >= min_h
            ]

            if not candidate_tables:
                return []

            # Inpaint / remove table lines for clean OCR
            cleaned_gray = gray.copy()
            cleaned_gray[table_mask > 0] = 255
            pil_img = Image.fromarray(cleaned_gray)

            # Extract words with coordinates via image_to_data
            data = pytesseract.image_to_data(
                pil_img,
                lang=settings.OCR_LANGUAGES,
                config="--psm 6",
                output_type=pytesseract.Output.DICT,
            )

            all_words = []
            for i in range(len(data["text"])):
                w_text = data["text"][i].strip()
                if w_text:
                    all_words.append({
                        "text": clean_arabic_text(w_text),
                        "x": data["left"][i],
                        "y": data["top"][i],
                        "w": data["width"][i],
                        "h": data["height"][i],
                    })

            for idx, (tx, ty, tw, th) in enumerate(candidate_tables):
                table_id = f"tbl_p{page_number}_{idx + 1}"

                # Words inside this table box
                tbl_words = [
                    w for w in all_words
                    if tx - 10 <= w["x"] <= tx + tw + 10 and ty - 10 <= w["y"] <= ty + th + 10
                ]

                if len(tbl_words) < 6:
                    continue

                # Detect vertical column divisions using vertical line projection
                sub_v = v_lines[ty : ty + th, tx : tx + tw]
                v_proj = np.sum(sub_v, axis=0)

                col_splits = [tx]
                for x in range(1, len(v_proj) - 1):
                    if v_proj[x] > (th * 255 * 0.25) and v_proj[x] >= v_proj[x - 1] and v_proj[x] >= v_proj[x + 1]:
                        abs_x = tx + x
                        if (abs_x - col_splits[-1]) > max(40, int(tw * 0.08)):
                            col_splits.append(abs_x)

                if (tx + tw - col_splits[-1]) > 40:
                    col_splits.append(tx + tw)

                # If columns found, form RTL column intervals (from right to left)
                cols = []
                if len(col_splits) >= 3:
                    for i in range(len(col_splits) - 1, 0, -1):
                        cols.append((col_splits[i - 1], col_splits[i]))
                else:
                    # Fallback to 4 proportional columns (الوجبة، الصنف، الكمية، التكرار)
                    col_w = tw / 4.0
                    for i in range(3, -1, -1):
                        cols.append((int(tx + i * col_w), int(tx + (i + 1) * col_w)))

                # Group words by Y-coordinates into rows
                tbl_words.sort(key=lambda item: item["y"])
                row_clusters = []
                current_cluster = []
                current_y = None
                row_tolerance = max(14, int(dpi * 0.08))

                for w in tbl_words:
                    if current_y is None or abs(w["y"] - current_y) <= row_tolerance:
                        current_cluster.append(w)
                        current_y = w["y"] if current_y is None else (current_y + w["y"]) / 2
                    else:
                        if current_cluster:
                            row_clusters.append(current_cluster)
                        current_cluster = [w]
                        current_y = w["y"]

                if current_cluster:
                    row_clusters.append(current_cluster)

                if len(row_clusters) < 2:
                    continue

                # Assemble grid rows by assigning words to columns
                grid_rows: list[list[str]] = []
                for cluster in row_clusters:
                    row_cells = []
                    for c_start, c_end in cols:
                        cell_words = [
                            w["text"]
                            for w in cluster
                            if (c_start - 12) <= (w["x"] + w["w"] / 2) <= (c_end + 12)
                        ]
                        row_cells.append(" ".join(cell_words).strip())

                    # Skip entirely empty rows
                    if any(cell for cell in row_cells):
                        grid_rows.append(row_cells)

                if len(grid_rows) < 2:
                    continue

                # Clean headers and rows
                raw_headers = grid_rows[0]
                headers = []
                default_names = ["الوجبة", "الصنف", "الكمية", "التكرار", "الملاحظات"]
                for c_idx, h in enumerate(raw_headers):
                    clean_h = h.strip()
                    if not clean_h:
                        clean_h = default_names[c_idx] if c_idx < len(default_names) else f"عمود_{c_idx+1}"
                    headers.append(clean_h)

                body_rows = grid_rows[1:]

                dict_rows: list[dict[str, str]] = []
                for r in body_rows:
                    row_dict = {}
                    for c_idx, col_name in enumerate(headers):
                        row_dict[col_name] = r[c_idx] if c_idx < len(r) else ""
                    dict_rows.append(row_dict)

                markdown_lines = [
                    "| " + " | ".join(headers) + " |",
                    "| " + " | ".join(["---"] * len(headers)) + " |",
                ]
                for r in body_rows:
                    markdown_lines.append("| " + " | ".join(r) + " |")

                scale = 72.0 / dpi
                pt_bbox = [tx * scale, ty * scale, (tx + tw) * scale, (ty + th) * scale]

                results.append(
                    ExtractedTable(
                        table_id=table_id,
                        page_number=page_number,
                        row_count=len(grid_rows),
                        col_count=len(headers),
                        headers=headers,
                        rows=body_rows,
                        bbox=pt_bbox,
                        markdown="\n".join(markdown_lines),
                        data_rows=dict_rows,
                    )
                )

        except Exception:
            pass

        return results
