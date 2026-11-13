import re
import unicodedata


def fix_reversed_arabic_line(line: str) -> str:
    """
    Detects if a line extracted from a PDF was stored in reversed visual order,
    and reverses it back to natural logical Arabic reading order while preserving
    English words and numbers in their proper LTR form.
    """
    if not line:
        return ""

    # Ensure spaces between numbers and Arabic letters before splitting
    normalized_line = re.sub(r"([\u0600-\u06FF])(\d+)", r"\1 \2", line)
    normalized_line = re.sub(r"(\d+)([\u0600-\u06FF])", r"\1 \2", normalized_line)

    words = normalized_line.split()
    if not words:
        return ""

    # Signals that text is in reversed visual order:
    # 1. Arabic words starting with Taa Marbuta (ة) - impossible in standard Arabic.
    # 2. Arabic words ending with Lam-Alif (لا / لأ / لإ) representing reversed prefix 'ال'.
    reversed_signals = sum(
        1
        for w in words
        if w.startswith("ة")
        or w.endswith("لا")
        or w.endswith("لأ")
        or w.endswith("لإ")
    )

    # Signals that text is already in natural logical order:
    logical_signals = sum(
        1
        for w in words
        if w.endswith("ة")
        or w.startswith("ال")
        or w.startswith("وال")
        or w.startswith("فال")
    )

    if reversed_signals > logical_signals and reversed_signals >= 1:
        fixed_words = []
        # Reverse word sequence in the line
        for w in reversed(words):
            # If word contains Arabic characters, reverse its letters
            if re.search(r"[\u0600-\u06FF]", w):
                fixed_words.append(w[::-1])
            else:
                # Keep numbers, dates (like 14/2026), Latin letters, and symbols unreversed
                fixed_words.append(w)
        return " ".join(fixed_words)

    return line


def is_corrupted_or_scanned_page(text: str, image_count: int = 0) -> bool:
    """
    Detects if a page has no text, is image-only, or has a corrupted digital layer
    (Mojibake / non-standard font encoding where Arabic text was extracted as scrambled ASCII symbols).
    """
    cleaned = text.strip()
    if len(cleaned) < 40:
        return True

    # Count real Arabic characters
    arabic_chars = len(re.findall(r"[\u0600-\u06FF]", cleaned))

    # If the page contains real Arabic characters, it's a valid digital layer
    if arabic_chars >= 25:
        return False

    # Check for symbol noise typical of broken font CMaps:
    weird_symbols = len(re.findall(r"[{}[\]|\\^~*#€@$%&;`<>]", cleaned))
    symbol_ratio = weird_symbols / len(cleaned)

    # If symbols are high or there are zero Arabic characters in an Arabic document, it must be OCR'd
    if symbol_ratio > 0.03 or arabic_chars == 0:
        return True

    return False


def clean_arabic_text(text: str) -> str:
    """
    Cleans and normalizes extracted Arabic and multilingual text:
    1. Normalizes unicode characters (NFKC).
    2. Detects and corrects reversed visual Arabic text layers.
    3. Removes invisible control characters (zero-width spaces, directional marks).
    4. Consolidates multiple blank lines and spaces while preserving paragraph breaks.
    5. Strips leading and trailing whitespaces.
    """
    if not text:
        return ""

    # Unicode NFKC normalization ensures combined characters/ligatures are standardized
    text = unicodedata.normalize("NFKC", text)

    # Remove unwanted control characters (except newline, carriage return, and tab)
    text = "".join(
        char
        for char in text
        if unicodedata.category(char)[0] != "C" or char in ("\n", "\r", "\t")
    )

    # Replace windows style line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Clean and check direction line-by-line
    lines = []
    for line in text.split("\n"):
        cleaned_line = re.sub(r"[ \t]+", " ", line).strip()
        if cleaned_line:
            # Fix reversed Arabic visual order if detected
            cleaned_line = fix_reversed_arabic_line(cleaned_line)
        lines.append(cleaned_line)

    # Recombine lines, avoiding more than two consecutive newlines
    cleaned_text = "\n".join(lines)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

    return cleaned_text.strip()
