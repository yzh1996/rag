"""Generate named fixtures for document-cleaning and PDF extraction tests."""

from __future__ import annotations

import html
import io
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import fitz
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "测试资源" / "文档清洗测试资料"
FONT = Path(r"C:\Windows\Fonts\msyh.ttc")
FONT_BOLD = Path(r"C:\Windows\Fonts\msyhbd.ttc")

if FONT.exists():
    pdfmetrics.registerFont(TTFont("FixtureSans", str(FONT)))
    pdfmetrics.registerFont(TTFont("FixtureSansBold", str(FONT_BOLD if FONT_BOLD.exists() else FONT)))
else:
    raise RuntimeError(f"中文字体不存在：{FONT}")


def _run(text: str) -> str:
    parts = text.split("\t")
    runs = []
    for index, part in enumerate(parts):
        if part:
            runs.append(f'<w:t xml:space="preserve">{escape(part)}</w:t>')
        if index < len(parts) - 1:
            runs.append("<w:tab/>")
    return "<w:r>" + "".join(runs) + "</w:r>"


def _paragraph(text: str, bold: bool = False) -> str:
    properties = '<w:rPr><w:b/></w:rPr>' if bold else ""
    return f"<w:p>{_run(text).replace('<w:r>', f'<w:r>{properties}', 1)}</w:p>"


def _part_xml(tag: str, text: str) -> bytes:
    return (
        f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:{tag} xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"{_paragraph(text)}"
        f"</w:{tag}>"
    ).encode("utf-8")


def _document_xml(paragraphs: list[str], header: bool, footer: bool) -> bytes:
    edge_refs = []
    if header:
        edge_refs.append('<w:headerReference w:type="default" r:id="rId1"/>')
    if footer:
        edge_refs.append('<w:footerReference w:type="default" r:id="rId2"/>')
    body = "".join(_paragraph(text, bold=index == 0) for index, text in enumerate(paragraphs))
    sect = (
        "<w:sectPr>"
        + "".join(edge_refs)
        + '<w:pgSz w:w="11906" w:h="16838"/>'
        + '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/>'
        + "</w:sectPr>"
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<w:body>{body}{sect}</w:body></w:document>"
    ).encode("utf-8")


def write_docx(path: Path, paragraphs: list[str], header: str | None = None, footer: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    has_header = header is not None
    has_footer = footer is not None
    overrides = [
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
    ]
    relationships = [
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
    ]
    document_relationships = []
    if has_header:
        overrides.append('<Override PartName="/word/header1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.header+xml"/>')
        document_relationships.append('<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/header" Target="header1.xml"/>')
    if has_footer:
        overrides.append('<Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/>')
        document_relationships.append('<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footer1.xml"/>')
    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        + "".join(overrides)
        + "</Types>"
    ).encode("utf-8")
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(relationships)
        + "</Relationships>"
    ).encode("utf-8")
    doc_rels = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(document_relationships)
        + "</Relationships>"
    ).encode("utf-8")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("word/document.xml", _document_xml(paragraphs, has_header, has_footer))
        archive.writestr("word/_rels/document.xml.rels", doc_rels)
        if has_header:
            archive.writestr("word/header1.xml", _part_xml("hdr", header))
        if has_footer:
            archive.writestr("word/footer1.xml", _part_xml("ftr", footer))


def _paragraph_style(name: str, size: float = 11, leading: float = 16) -> ParagraphStyle:
    return ParagraphStyle(name, fontName="FixtureSans", fontSize=size, leading=leading, textColor=colors.HexColor("#1F2933"))


def write_complex_pdf(path: Path) -> None:
    width, height = A4
    canvas = Canvas(str(path), pagesize=A4)
    canvas.setFillColor(colors.HexColor("#0C3559"))
    canvas.rect(0, height - 82, width, 82, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("FixtureSansBold", 18)
    canvas.drawString(36, height - 38, "复杂排版测试")
    canvas.setFont("FixtureSans", 8)
    canvas.drawString(36, height - 60, "页眉区域 | DOCUMENT PROCESSING FIXTURE")
    canvas.setFillColor(colors.HexColor("#0C3559"))
    canvas.setFont("FixtureSansBold", 17)
    canvas.drawString(36, height - 122, "企业知识库文档处理综合样本")
    intro = Paragraph(
        html.escape("本页同时包含页眉、分栏正文、指标卡片和表格，用来检查 PDF 文本提取后的阅读顺序与内容完整性。"),
        _paragraph_style("complex-intro", 10.5, 15),
    )
    intro.wrapOn(canvas, 250, 100)
    intro.drawOn(canvas, 36, height - 245)
    canvas.setFillColor(colors.HexColor("#E4F0F8"))
    canvas.setStrokeColor(colors.HexColor("#3984A9"))
    canvas.roundRect(320, height - 245, 238, 112, 6, fill=1, stroke=1)
    canvas.setFillColor(colors.HexColor("#11466B"))
    canvas.setFont("FixtureSansBold", 12)
    canvas.drawString(337, height - 162, "关键指标")
    for row, (label, value) in enumerate((("正文区域", "2"), ("表格行数", "3"), ("处理状态", "可用"))):
        y = height - 188 - row * 22
        canvas.setFont("FixtureSans", 9.5)
        canvas.setFillColor(colors.HexColor("#26323B"))
        canvas.drawString(337, y, label)
        canvas.setFont("FixtureSansBold", 10)
        canvas.setFillColor(colors.HexColor("#0D4C77"))
        canvas.drawRightString(530, y, value)
    canvas.setFillColor(colors.HexColor("#0C3559"))
    canvas.setFont("FixtureSansBold", 13)
    canvas.drawString(36, height - 290, "流程信息")
    table_data = [
        ["编号", "处理步骤", "输出检查"],
        ["PDF-001", "复杂排版提取", "保留标题与正文"],
        ["PDF-002", "布局内容解析", "保留指标卡片"],
        ["PDF-003", "结果清洗", "去除多余空白"],
    ]
    table = Table(table_data, colWidths=[90, 170, 220], rowHeights=[28, 36, 36, 36])
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "FixtureSans"),
        ("FONTNAME", (0, 0), (-1, 0), "FixtureSansBold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D6EAF4")),
        ("GRID", (0, 0), (-1, -1), 0.6, colors.HexColor("#8EA0AD")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    table.wrapOn(canvas, width, height)
    table.drawOn(canvas, 36, height - 495)
    summary = Paragraph(
        html.escape("页脚区域保留测试编号，正文区域用于验证复杂布局中的文字顺序。"),
        _paragraph_style("complex-summary", 10.5, 15),
    )
    summary.wrapOn(canvas, width - 72, 80)
    summary.drawOn(canvas, 36, 115)
    canvas.setFillColor(colors.HexColor("#59636B"))
    canvas.setFont("FixtureSans", 8)
    canvas.drawString(36, 35, "复杂排版测试 | 第 1 页")
    canvas.save()


def write_table_pdf(path: Path) -> None:
    width, height = A4
    canvas = Canvas(str(path), pagesize=A4)
    canvas.setFillColor(colors.HexColor("#173C5C"))
    canvas.setFont("FixtureSansBold", 20)
    canvas.drawString(42, height - 72, "表格测试")
    intro = Paragraph(
        html.escape("下面的表格用于验证行列关系、中文单元格和表头是否能被文档处理器提取。"),
        _paragraph_style("table-intro", 11, 16),
    )
    intro.wrapOn(canvas, width - 84, 60)
    intro.drawOn(canvas, 42, height - 135)
    rows = [
        ["编号", "流程阶段", "责任人", "处理要求", "状态"],
        ["KB-001", "文档入库", "知识管理员", "核对文件名称、分类和来源", "启用"],
        ["KB-002", "内容解析", "平台服务", "提取正文、表格及页面信息", "启用"],
        ["KB-003", "质量复核", "业务负责人", "确认版本、生效日期和责任部门", "待复核"],
    ]
    table = Table(rows, colWidths=[68, 100, 96, 190, 70], rowHeights=[34, 54, 54, 54])
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "FixtureSans"),
        ("FONTNAME", (0, 0), (-1, 0), "FixtureSansBold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D4E8F2")),
        ("GRID", (0, 0), (-1, -1), 0.7, colors.HexColor("#8796A0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    table.wrapOn(canvas, width, height)
    table.drawOn(canvas, 42, height - 470)
    canvas.setFillColor(colors.HexColor("#555E66"))
    canvas.setFont("FixtureSans", 9)
    canvas.drawString(42, 40, "表格测试 | 需要保留表头、行内容和页码信息")
    canvas.save()


def write_scanned_pdf(path: Path) -> None:
    image = Image.new("RGB", (1654, 2339), "white")
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.truetype(str(FONT_BOLD if FONT_BOLD.exists() else FONT), 58)
    body_font = ImageFont.truetype(str(FONT), 34)
    draw.text((145, 135), "扫描件测试", font=title_font, fill=(19, 43, 62))
    lines = [
        "这是没有文字层的扫描件页面。",
        "文档处理器需要先识别图片中的文字，",
        "再继续执行文本清洗和分段。",
        "测试编号：OCR-001。",
    ]
    y = 320
    for line in lines:
        draw.text((155, y), line, font=body_font, fill=(30, 30, 30))
        y += 70
    draw.text((155, y + 40), "扫描件测试页：文字来自图片，不包含可直接复制的文本层。", font=body_font, fill=(35, 35, 35))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85, optimize=True)
    document = fitz.open()
    page = document.new_page(width=595, height=842)
    page.insert_image(page.rect, stream=buffer.getvalue())
    document.save(path)
    document.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_docx(
        OUT / "页眉清洗测试.docx",
        ["页眉清洗测试", "正文内容应当保留，标题信息不能丢失。", "清洗后仍应能够看到这段正文。"],
        header="页眉：企业知识库文档处理测试",
    )
    write_docx(
        OUT / "免责声明清洗测试.docx",
        ["免责声明清洗测试", "正文内容：这是一段需要保留的业务说明。", "免责声明：本文档仅用于文档加载器测试，不构成正式业务依据。", "说明结束，正文需要继续保留。"],
    )
    write_docx(
        OUT / "文本规范化测试.docx",
        ["文本规范化测试", "  多余空格    与\t制表符应被统一。", "", "", "第二段内容应该保留，并且空行数量应当收敛。"],
    )
    write_complex_pdf(OUT / "复杂排版测试.pdf")
    write_table_pdf(OUT / "表格测试.pdf")
    write_scanned_pdf(OUT / "扫描件测试.pdf")
    print(f"Generated {len(list(OUT.iterdir()))} fixtures in {OUT}")


if __name__ == "__main__":
    main()
