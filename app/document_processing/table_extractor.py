import re
from typing import Optional
import pymupdf
from app.document_processing.cleaner import clean_arabic_text
from app.schemas.document import ExtractedTable


class TableExtractor:
    """
    Detects and extracts structured tables from PDF pages using PyMuPDF table detection.
    Normalizes Arabic cell contents, handles reversed fonts, and formats tabular data
    into structured rows, Markdown format, and JSON-ready dicts for ERP ingestion.
    """

    @classmethod
    def extract_tables_from_page(
        cls,
        page: pymupdf.Page,
        page_number: int,
    ) -> list[ExtractedTable]:
        """
        Searches the page for vector and text-aligned tables, extracts rows/columns,
        and returns a list of ExtractedTable schemas.
        """
        extracted_tables: list[ExtractedTable] = []

        try:
            # PyMuPDF built-in geometric table finder (use_layout=False for fast native vector analysis)
            tabs = page.find_tables(use_layout=False)
            if not tabs or not tabs.tables:
                return []

            for idx, tab in enumerate(tabs.tables):
                table_id = f"tbl_p{page_number}_{idx + 1}"
                raw_data = tab.extract()  # Returns list of rows (each row is list of str / None)

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

                # Identify header vs rows
                headers = [h if h else f"عمود_{i+1}" for i, h in enumerate(cleaned_grid[0])]
                data_rows_grid = cleaned_grid[1:] if len(cleaned_grid) > 1 else []

                # Build data_rows dictionary list
                dict_rows: list[dict[str, str]] = []
                for row in data_rows_grid:
                    row_dict = {}
                    for col_idx, col_name in enumerate(headers):
                        val = row[col_idx] if col_idx < len(row) else ""
                        row_dict[col_name] = val
                    dict_rows.append(row_dict)

                # Format Markdown representation
                markdown_lines = []
                header_line = "| " + " | ".join(headers) + " |"
                separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
                markdown_lines.append(header_line)
                markdown_lines.append(separator_line)

                for r in data_rows_grid:
                    row_line = "| " + " | ".join(r) + " |"
                    markdown_lines.append(row_line)

                markdown_str = "\n".join(markdown_lines)

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
                        markdown=markdown_str,
                        data_rows=dict_rows,
                    )
                )

        except Exception:
            # Non-blocking graceful fallback
            return []

        return extracted_tables
