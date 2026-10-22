"""Generate knowledge-base fixture documents in several formats."""

from __future__ import annotations

import csv
import html
import io
import re
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import fitz
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "knowledge_samples"
FONT = Path(r"C:\Windows\Fonts\msyh.ttc")
FONT_BOLD = Path(r"C:\Windows\Fonts\msyhbd.ttc")
pdfmetrics.registerFont(TTFont("KnowledgeSans", str(FONT)))
pdfmetrics.registerFont(TTFont("KnowledgeSansBold", str(FONT_BOLD)))

TITLE = "企业知识库与文档处理规范"
BODY = (
    "企业知识库用于集中保存制度、流程、产品资料和项目经验，并通过文档解析、分段、向量化与检索帮助员工快速找到可靠信息。"
    "文档上传后，系统会识别文件类型，提取正文和表格内容，补充来源、类别与上传时间等元数据，再按照固定长度切分文本。"
    "使用者提问时，检索器会先召回相关片段，再交给问答模型结合上下文生成回答。为了保证结果稳定，资料应使用清晰标题，避免把多个主题混在同一段中。"
    "涉及制度变更时，应保留版本号、生效日期和责任部门；发现重复或过期内容，应及时合并、标记或移除。"
)
BODY = BODY[:300]
TABLE_ROWS = [
    ("文档入库", "上传文件并选择分类", "记录来源、文件名和时间"),
    ("内容解析", "提取文本、表格和页面信息", "检查编码与空内容"),
    ("分段向量化", "按段落切分并写入向量库", "保留原文位置元数据"),
    ("检索问答", "召回相关片段并生成答案", "回答应引用知识库内容"),
]


def write_text_files() -> None:
    (OUT / "knowledge_sample.txt").write_text(
        f"{TITLE}\n\n{BODY}\n", encoding="utf-8", newline="\n"
    )
    md = (
        f"# {TITLE}\n\n"
        f"{BODY}\n\n"
        "## 使用规范\n\n"
        "- 标注来源、版本和生效日期。\n"
        "- 定期清理重复与过期资料。\n"
    )
    (OUT / "knowledge_sample.md").write_text(md, encoding="utf-8", newline="\n")


def write_csv() -> None:
    rows = [
        ["编号", "流程阶段", "负责人", "处理要求", "状态"],
        ["KB-001", "文档入库", "知识管理员", "核对文件名、分类和来源", "启用"],
        ["KB-002", "内容解析", "平台服务", "提取正文、表格及页面信息", "启用"],
        ["KB-003", "分段向量化", "平台服务", "按段落切分并保存元数据", "启用"],
        ["KB-004", "质量复核", "业务负责人", "确认版本、生效日期和责任部门", "启用"],
        ["KB-005", "检索问答", "问答服务", "召回相关片段并生成可追溯回答", "启用"],
        ["KB-006", "资料维护", "知识管理员", "合并重复资料并标记旧版本", "启用"],
    ]
    with (OUT / "knowledge_sample.csv").open("w", encoding="utf-8-sig", newline="") as f:
        csv.writer(f).writerows(rows)


def xml_document(body_xml: str) -> bytes:
    document = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>{body_xml}<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/></w:sectPr></w:body>
</w:document>'''
    return document.encode("utf-8")


def wp(text: str, *, bold: bool = False, size: int = 24) -> str:
    props = f'<w:rPr><w:sz w:val="{size}"/>'
    if bold:
        props += "<w:b/>"
    props += "</w:rPr>"
    return f"<w:r>{props}<w:t xml:space=\"preserve\">{escape(text)}</w:t></w:r>"


def paragraph(text: str, *, bold: bool = False, size: int = 24) -> str:
    return f"<w:p><w:pPr><w:spacing w:after=\"160\"/></w:pPr>{wp(text, bold=bold, size=size)}</w:p>"


def write_docx() -> None:
    body_xml = paragraph(TITLE, bold=True, size=32) + paragraph(BODY)
    body_xml += paragraph("规范：标注来源、版本和生效日期，并定期清理重复资料。")
    content_types = b'''<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>'''
    rels = b'''<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>'''
    with zipfile.ZipFile(OUT / "knowledge_sample.docx", "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("word/document.xml", xml_document(body_xml))


def page_text_pdf(path: Path, title: str, body: str, header: str | None = None) -> None:
    c = Canvas(str(path), pagesize=A4)
    width, height = A4
    c.setFillColor(colors.HexColor("#15385F"))
    c.setFont("KnowledgeSansBold", 22)
    c.drawString(52, height - 72, title)
    style = ParagraphStyle("body", fontName="KnowledgeSans", fontSize=12, leading=18, textColor=colors.HexColor("#202020"))
    paragraph = Paragraph(html.escape(body), style)
    paragraph.wrapOn(c, width - 104, height - 170)
    paragraph.drawOn(c, 52, 125)
    c.setFillColor(colors.HexColor("#666666"))
    c.setFont("KnowledgeSans", 9)
    if header:
        c.drawString(52, height - 32, header)
    c.drawString(52, 32, "企业知识库测试资料")
    c.save()


def table_pdf(path: Path) -> None:
    c = Canvas(str(path), pagesize=A4)
    width, height = A4
    c.setFillColor(colors.HexColor("#0D3D66"))
    c.setFont("KnowledgeSansBold", 19)
    c.drawString(45, height - 65, "知识库处理流程与责任矩阵")
    intro = Paragraph(html.escape(BODY[:120]), ParagraphStyle("intro", fontName="KnowledgeSans", fontSize=10.5, leading=15))
    intro.wrapOn(c, width - 90, 60)
    intro.drawOn(c, 45, height - 145)
    data = [["编号", "流程阶段", "负责人", "处理要求"]]
    data += [[f"KB-{i:03d}", row[0], row[1], row[2]] for i, row in enumerate(TABLE_ROWS[:3], 1)]
    table = Table(data, colWidths=[70, 120, 110, 205], rowHeights=[34] + [62] * 3)
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "KnowledgeSans"),
        ("FONTNAME", (0, 0), (-1, 0), "KnowledgeSansBold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D2E8F4")),
        ("GRID", (0, 0), (-1, -1), 0.6, colors.HexColor("#8C9BA5")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    table.wrapOn(c, width, height)
    table.drawOn(c, 45, height - 410)
    rest = Paragraph(html.escape("资料变更时保留版本号和生效日期，便于追溯。"), ParagraphStyle("rest", fontName="KnowledgeSans", fontSize=11, leading=16))
    rest.wrapOn(c, width - 90, 190)
    rest.drawOn(c, 45, 125)
    c.save()


def scanned_pdf(path: Path) -> None:
    image = Image.new("RGB", (1654, 2339), "white")
    draw = ImageDraw.Draw(image)
    font_title = ImageFont.truetype(str(FONT), 58)
    font_body = ImageFont.truetype(str(FONT), 34)
    draw.text((150, 140), TITLE, font=font_title, fill=(20, 35, 48))
    y = 300
    words = re.findall(r".{1,24}", BODY)
    for line in words:
        draw.text((155, y), line, font=font_body, fill=(25, 25, 25))
        y += 62
    draw.text((155, y + 45), "扫描件测试页：文字来自图片，不包含可直接复制的文本层。", font=font_body, fill=(35, 35, 35))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_image(page.rect, stream=buf.getvalue())
    doc.save(path)
    doc.close()


def complex_pdf(path: Path) -> None:
    c = Canvas(str(path), pagesize=A4)
    width, height = A4
    c.setFillColor(colors.HexColor("#F6F9FB")); c.rect(0, 0, width, height, fill=1, stroke=0)
    c.setFillColor(colors.HexColor("#0D3359")); c.rect(0, height - 95, width, 95, fill=1, stroke=0)
    c.setFillColor(colors.white); c.setFont("KnowledgeSansBold", 15); c.drawString(42, height - 48, "企业知识库 | 版式综合测试")
    c.setFont("KnowledgeSans", 8); c.setFillColor(colors.HexColor("#C4D9E8")); c.drawString(42, height - 78, "DOCUMENT PROCESSING BRIEF")
    c.setFillColor(colors.HexColor("#0D3359")); c.setFont("KnowledgeSansBold", 21); c.drawString(42, height - 145, TITLE)
    left = Paragraph(html.escape(BODY[:110]), ParagraphStyle("left", fontName="KnowledgeSans", fontSize=10.5, leading=15))
    left.wrapOn(c, 243, 260); left.drawOn(c, 42, height - 420)
    c.setFillColor(colors.HexColor("#DEEFF8")); c.setStrokeColor(colors.HexColor("#408CB3")); c.roundRect(315, height - 330, 238, 160, 6, fill=1, stroke=1)
    c.setFillColor(colors.HexColor("#0D4168")); c.setFont("KnowledgeSansBold", 14); c.drawString(334, height - 205, "关键指标")
    for i, (label, value) in enumerate([("文档片段", "7"), ("检索 Top K", "5"), ("版本状态", "有效")]):
        yy = height - 245 - i * 28; c.setFont("KnowledgeSans", 10); c.setFillColor(colors.HexColor("#26323B")); c.drawString(334, yy, label)
        c.setFont("KnowledgeSansBold", 11); c.setFillColor(colors.HexColor("#0D4C77")); c.drawRightString(530, yy, value)
    c.setFillColor(colors.HexColor("#0D3359")); c.setFont("KnowledgeSansBold", 15); c.drawString(42, height - 485, "流程摘要")
    c.setStrokeColor(colors.HexColor("#408CB3")); c.line(42, height - 497, 553, height - 497)
    summary = Paragraph(html.escape(BODY[110:145] + " 系统记录来源和处理日志，便于审计。"), ParagraphStyle("summary", fontName="KnowledgeSans", fontSize=11, leading=16))
    summary.wrapOn(c, width - 84, 150); summary.drawOn(c, 42, 160)
    c.setFillColor(colors.HexColor("#404B55")); c.setFont("KnowledgeSans", 8.5); c.drawString(42, 37, "机密级别：内部使用    |    页码 1 / 1")
    c.save()


def cleaning_pdf(path: Path) -> None:
    c = Canvas(str(path), pagesize=A4)
    width, height = A4
    header = "知识库清洗测试页   |   Internal Knowledge Base   |   Internal Knowledge Base"
    c.setFillColor(colors.HexColor("#5A5A5A")); c.setFont("KnowledgeSans", 8.5); c.drawString(40, height - 31, header)
    c.setStrokeColor(colors.HexColor("#999999")); c.line(40, height - 40, width - 40, height - 40)
    c.setFillColor(colors.HexColor("#202020")); c.setFont("KnowledgeSansBold", 19); c.drawString(45, height - 78, "文档清洗与去重测试材料")
    noisy = (
        "  企业知识库需要在入库前清理原始资料。   文档解析服务会识别标题、正文和元数据，删除多余空格，统一换行，保留必要的版本信息。 "
        "企业知识库需要在入库前清理原始资料。\n\n"
        + BODY
        + "\n\n【测试符号】 @@@ ### $$$ %% ^^^   ① ② ③   中文与 English 混排。\n"
        + "企业知识库需要在入库前清理原始资料。   文档解析服务会识别标题、正文和元数据，删除多余空格，统一换行，保留必要的版本信息。"
    )
    style = ParagraphStyle("noisy", fontName="KnowledgeSans", fontSize=11.5, leading=17, textColor=colors.HexColor("#202020"))
    paragraph = Paragraph(html.escape(noisy).replace("\n", "<br/>") , style)
    paragraph.wrapOn(c, width - 90, height - 190); paragraph.drawOn(c, 45, 92)
    c.setFillColor(colors.HexColor("#5A5A5A")); c.setFont("KnowledgeSans", 8.5); c.drawString(45, 37, "页脚重复内容：知识库清洗测试材料   知识库清洗测试材料")
    c.save()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_text_files()
    write_csv()
    write_docx()
    page_text_pdf(OUT / "knowledge_sample_text.pdf", TITLE, BODY)
    table_pdf(OUT / "knowledge_sample_table.pdf")
    scanned_pdf(OUT / "knowledge_sample_scanned.pdf")
    complex_pdf(OUT / "knowledge_sample_complex_layout.pdf")
    cleaning_pdf(OUT / "knowledge_sample_cleaning_dedup.pdf")
    print(f"Generated {len(list(OUT.iterdir()))} files in {OUT}")


if __name__ == "__main__":
    main()
