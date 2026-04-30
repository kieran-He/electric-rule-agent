from __future__ import annotations

from io import BytesIO
from pathlib import Path

from dataprocess.schemas import RawPage


class DocumentParseError(RuntimeError):
    pass


def _parse_with_pdfplumber(path: Path) -> list[RawPage]:
    import pdfplumber

    pages: list[RawPage] = []
    with pdfplumber.open(path) as pdf:
        for index, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append(RawPage(page_number=index, text=text))
    return pages


def _parse_with_pymupdf(path: Path) -> list[RawPage]:
    import fitz

    pages: list[RawPage] = []
    document = fitz.open(path)
    try:
        for index, page in enumerate(document, start=1):
            pages.append(RawPage(page_number=index, text=page.get_text("text")))
    finally:
        document.close()
    return pages


def parse_pdf(file_path: str) -> list[RawPage]:
    path = Path(file_path)
    errors: list[str] = []
    for parser in (_parse_with_pdfplumber, _parse_with_pymupdf):
        try:
            pages = parser(path)
            if any(page.text.strip() for page in pages):
                return pages
        except Exception as exc:
            errors.append(f"{parser.__name__}: {exc}")
    raise DocumentParseError(f"Failed to parse PDF {file_path}. Details: {' | '.join(errors)}")


def parse_pdf_ocr(
    file_path: str,
    lang: str = "chi_sim+eng",
    dpi: int = 300,
    tesseract_cmd: str | None = None,
    tessdata_dir: str | None = None,
) -> list[RawPage]:
    path = Path(file_path)
    try:
        import fitz
        import pytesseract
        from PIL import Image
    except Exception as exc:
        raise DocumentParseError(f"OCR dependencies unavailable for {file_path}: {exc}") from exc

    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    pages: list[RawPage] = []
    try:
        document = fitz.open(path)
        zoom = max(dpi / 72.0, 1.0)
        matrix = fitz.Matrix(zoom, zoom)
        try:
            for index, page in enumerate(document, start=1):
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                image = Image.open(BytesIO(pix.tobytes("png")))
                config = "--psm 6"
                if tessdata_dir:
                    normalized_tessdata_dir = tessdata_dir.replace("\\", "/")
                    config = f"{config} --tessdata-dir {normalized_tessdata_dir}"
                text = pytesseract.image_to_string(image, lang=lang, config=config)
                pages.append(RawPage(page_number=index, text=text or ""))
        finally:
            document.close()
    except Exception as exc:
        raise DocumentParseError(f"Failed to OCR PDF {file_path}: {exc}") from exc

    if not any(page.text.strip() for page in pages):
        raise DocumentParseError(f"OCR returned empty text for {file_path}")
    return pages