"""PDF逐页提取、扫描页OCR与保守文档清洗。"""
import re
from collections import Counter

from langchain_core.documents import Document

from config.settings import settings


def _ocr_page_tesseract(page):
    try:
        textpage = page.get_textpage_ocr(language=settings.OCR_LANGUAGE, dpi=settings.OCR_DPI, full=True)
        return page.get_text("text", textpage=textpage)
    except Exception as exc:
        raise ValueError("Tesseract识别失败，请检查Tesseract、TESSDATA_PREFIX和OCR语言包") from exc


def _ocr_page_tencent(page):
    from core.tencent_ocr import recognize_image

    image = page.get_pixmap(dpi=settings.OCR_DPI, alpha=False).tobytes("png")
    return recognize_image(image)


def ocr_page(page):
    if not settings.OCR_ENABLED:
        raise ValueError("检测到扫描页，请启用OCR后重新处理")
    try:
        if settings.OCR_PROVIDER == "tesseract":
            return _ocr_page_tesseract(page)
        if settings.OCR_PROVIDER == "tencent":
            return _ocr_page_tencent(page)
        raise ValueError("OCR_PROVIDER只支持tencent或tesseract")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("扫描页OCR服务暂时不可用，请检查网络或稍后重试") from exc

def _create_pdf_loader(path, loader_name=None):
    from langchain_community.document_loaders import PDFPlumberLoader, PyMuPDFLoader

    key = (loader_name or "pymupdf").strip().lower().replace("_", "").replace("-", "")
    if key in {"pymupdf", "pymupdfloader", "fitz"}:
        loader_class = PyMuPDFLoader
    elif key in {"pdfplumber", "pdfplumberloader"}:
        loader_class = PDFPlumberLoader
    else:
        raise ValueError(f"不支持的PDF加载器：{loader_name}")
    return loader_class(str(path)), loader_class.__name__


def _table_to_markdown(rows):
    if not rows or not any(cell is not None and str(cell).strip() for row in rows for cell in row):
        return ""
    width = max(len(row) for row in rows)
    lines = []
    for row in rows:
        cells = []
        for cell in list(row) + [None] * (width - len(row)):
            text = "" if cell is None else str(cell)
            text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
            cells.append(text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>"))
        lines.append("| " + " | ".join(cells) + " |")
    lines.insert(1, "| " + " | ".join(["---"] * width) + " |")
    return "\n".join(lines)


def _convert_pdf_tables_to_markdown(path, documents):
    """按位置替换表格文字，保留表格外的正文，避免重复提取。"""
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        for page, document in zip(pdf.pages, documents):
            tables = []
            for table in page.find_tables():
                markdown = _table_to_markdown(table.extract())
                if markdown:
                    tables.append((table.bbox, markdown))
            if not tables:
                continue

            def outside_tables(obj):
                if obj.get("object_type") != "char":
                    return True
                x = (obj["x0"] + obj["x1"]) / 2
                y = (obj["top"] + obj["bottom"]) / 2
                return not any(
                    x0 <= x <= x1 and top <= y <= bottom
                    for (x0, top, x1, bottom), _ in tables
                )

            lines = page.filter(outside_tables).extract_text_lines(return_chars=False)
            blocks = [(line["top"], line["x0"], line["text"]) for line in lines]
            blocks.extend((bbox[1], bbox[0], "\n" + markdown + "\n") for bbox, markdown in tables)
            document.page_content = "\n".join(
                text for _, _, text in sorted(blocks, key=lambda block: (block[0], block[1]))
            ).strip()


def extract_pdf(path, loader_name=None):
    import fitz

    with fitz.open(path) as pdf:
        if pdf.is_encrypted:
            raise ValueError("加密PDF请先解密再上传")
        if len(pdf) > settings.MAX_PDF_PAGES:
            raise ValueError(f"PDF超过{settings.MAX_PDF_PAGES}页上限")
        image_pages = [bool(page.get_images()) for page in pdf]

    loader, loader_class_name = _create_pdf_loader(path, loader_name)
    loaded_documents = loader.load()
    result = []
    for number, has_image in enumerate(image_pages, 1):
        source = loaded_documents[number - 1] if number <= len(loaded_documents) else Document(page_content="", metadata={})
        text = source.page_content or ""
        scanned = len(re.sub(r"\s", "", text)) < settings.OCR_MIN_TEXT_CHARS and has_image
        metadata = {
            **source.metadata,
            "page": number,
            "ocr_required": scanned,
            "ocr_used": False,
            "loader": loader_class_name,
        }
        result.append(Document(page_content=text, metadata=metadata))
    if loader_class_name == "PDFPlumberLoader":
        _convert_pdf_tables_to_markdown(path, result)
    return result


def recognize_documents(path, documents):
    """只对检测到的扫描页执行OCR，保留其他页的提取结果。"""
    result = [Document(page_content=doc.page_content, metadata=dict(doc.metadata)) for doc in documents]
    if not any(doc.metadata.get("ocr_required") for doc in result):
        return result
    import fitz
    with fitz.open(path) as pdf:
        for doc in result:
            if not doc.metadata.get("ocr_required"):
                continue
            number = doc.metadata["page"]
            try:
                doc.page_content = ocr_page(pdf[number - 1])
            except ValueError as exc:
                raise ValueError(f"第{number}页OCR失败：{exc}") from exc
            if not doc.page_content.strip():
                raise ValueError(f"第{number}页扫描件未识别出文字，请人工检查")
            doc.metadata["ocr_used"] = True
    return result


def load_pdf(path, loader_name=None):
    return recognize_documents(path, extract_pdf(path, loader_name=loader_name))


def clean_documents(documents):
    if not settings.DOCUMENT_CLEANING_ENABLED:
        return [doc for doc in documents if doc.page_content.strip()]
    result = []
    for doc in documents:
        metadata = dict(doc.metadata)
        header_text = metadata.pop("_docx_header_prefix", "")
        text = doc.page_content
        if header_text and text.startswith(header_text):
            text = text[len(header_text):]
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
        text = "\n".join(
            line for line in text.splitlines()
            if not re.match(r"^\s*免责声明\s*[:：]\s*\S+", line)
        )
        text = "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines())
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if text:
            result.append(Document(page_content=text, metadata={**metadata, "cleaned": True}))
    pages = [doc for doc in result if "page" in doc.metadata]
    if settings.REMOVE_REPEATED_PAGE_EDGES and len(pages) >= 3:
        repeated = {}
        for position in (0, -1):
            counts = Counter(doc.page_content.splitlines()[position] for doc in pages)
            repeated[position] = {line for line, count in counts.items() if count >= max(3, len(pages) * 0.8) and len(line) <= 100}
        for doc in pages:
            lines = doc.page_content.splitlines()
            if len(lines) > 2:
                start = 1 if lines[0] in repeated[0] else 0
                end = len(lines) - 1 if lines[-1] in repeated[-1] else len(lines)
                doc.page_content = "\n".join(lines[start:end]).strip()
    return result
