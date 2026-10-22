"""文档清洗与 PDF 处理效果测试。每个测试都会生成可打开的前后对比产物。"""

from __future__ import annotations

import json
from pathlib import Path
from html import escape

import pytest

from core.content_processing import clean_documents, recognize_documents
from core.document_loader import DocumentLoader
from config.settings import settings

FIXTURES = Path(__file__).resolve().parent / "测试资源" / "文档清洗测试资料"
ARTIFACTS = Path(__file__).resolve().parent / "测试产物" / "文档清洗"


def _load(name: str, loader_name: str | None = None):
    path = FIXTURES / name
    raw = DocumentLoader().load_raw_file(str(path), loader_name=loader_name)
    recognized = recognize_documents(str(path), raw)
    before = "\n\n".join(doc.page_content for doc in recognized)
    cleaned = clean_documents(recognized)
    after = "\n\n".join(doc.page_content for doc in cleaned)
    return path, recognized, cleaned, before, after


def _write_artifacts(case_name: str, path: Path, before: str, after: str, documents) -> None:
    target = ARTIFACTS / case_name
    target.mkdir(parents=True, exist_ok=True)
    (target / "清洗前.txt").write_text(before, encoding="utf-8")
    (target / "清洗后.txt").write_text(after, encoding="utf-8")
    metadata = [doc.metadata for doc in documents]
    (target / "结果.json").write_text(json.dumps({
        "文件": path.name,
        "页数或文档段数": len(documents),
        "清洗前字符数": len(before),
        "清洗后字符数": len(after),
        "使用的加载器": next((item.get("loader") for item in metadata if item.get("loader")), "未知"),
        "文档元数据": metadata,
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    html = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>{escape(case_name)}清洗对比</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px}}.panes{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}section{{border:1px solid #ccd3da;border-radius:8px;padding:16px}}pre{{white-space:pre-wrap;overflow:auto;max-height:720px;line-height:1.6}}</style>
<h1>{escape(case_name)}：{escape(path.name)}</h1><div class="panes"><section><h2>清洗前</h2><pre>{escape(before)}</pre></section><section><h2>清洗后</h2><pre>{escape(after)}</pre></section></div></html>'''
    (target / "对比.html").write_text(html, encoding="utf-8")
    loader_name = next((item.metadata.get("loader") for item in documents if item.metadata.get("loader")), "未知")
    print(f"[{case_name}] 加载器：{loader_name}；清洗前 {len(before)} 字符 -> 清洗后 {len(after)} 字符；对比文件：{target / '对比.html'}")


class TestDocumentCleaning:
    """每个方法都可以在 IDE 中右键单独运行。"""

    def test_页眉清洗(self):
        path, _, cleaned, before, after = _load("页眉清洗测试.docx")
        _write_artifacts("页眉清洗", path, before, after, cleaned)
        assert "页眉：企业知识库文档处理测试" in before
        assert "页眉：企业知识库文档处理测试" not in after
        assert "正文内容应当保留" in after
        assert cleaned[0].metadata["loader"] == "Docx2txtLoader"

    def test_免责声明清洗(self):
        path, _, cleaned, before, after = _load("免责声明清洗测试.docx")
        _write_artifacts("免责声明清洗", path, before, after, cleaned)
        assert "免责声明：" in before
        assert "免责声明：" not in after
        assert "说明结束，正文需要继续保留" in after
        assert cleaned[0].metadata["loader"] == "Docx2txtLoader"

    def test_文本规范化(self):
        path, _, cleaned, before, after = _load("文本规范化测试.docx")
        _write_artifacts("文本规范化", path, before, after, cleaned)
        assert "    " in before
        assert "    " not in after
        assert "\n\n\n" in before
        assert "\n\n\n" not in after
        assert cleaned[0].metadata["loader"] == "Docx2txtLoader"

    def test_pdf复杂排版(self):
        path, _, cleaned, before, after = _load("复杂排版测试.pdf", loader_name="pymupdf")
        _write_artifacts("PDF复杂排版", path, before, after, cleaned)
        assert len(before.strip()) > 100
        assert "PDF-001" in after
        assert len(cleaned) == 1
        assert cleaned[0].metadata["loader"] == "PyMuPDFLoader"

    def test_pdf表格(self):
        path, _, cleaned, before, after = _load("表格测试.pdf", loader_name="pdfplumber")
        _write_artifacts("PDF表格", path, before, after, cleaned)
        (ARTIFACTS / "PDF表格" / "清洗后.md").write_text(after, encoding="utf-8")
        expected_table = "\n".join([
            "| 编号 | 流程阶段 | 责任人 | 处理要求 | 状态 |",
            "| --- | --- | --- | --- | --- |",
            "| KB-001 | 文档入库 | 知识管理员 | 核对文件名称、分类和来源 | 启用 |",
            "| KB-002 | 内容解析 | 平台服务 | 提取正文、表格及页面信息 | 启用 |",
            "| KB-003 | 质量复核 | 业务负责人 | 确认版本、生效日期和责任部门 | 待复核 |",
        ])
        for text in (before, after):
            assert expected_table in text
            assert "下面的表格用于验证行列关系、中文单元格和表头是否能被文档处理器提取。" in text
            assert text.index("下面的表格") < text.index("| 编号 |") < text.index("表格测试 | 需要保留")
            for identifier in ("KB-001", "KB-002", "KB-003"):
                assert text.count(identifier) == 1
        assert len(cleaned) == 1
        assert cleaned[0].metadata["loader"] == "PDFPlumberLoader"

    def test_pdf表格_特殊单元格和多页(self, tmp_path):
        import fitz
        from langchain_community.document_loaders import PDFPlumberLoader

        path = tmp_path / "table_cells.pdf"
        with fitz.open() as pdf:
            page = pdf.new_page(width=520, height=420)

            def draw_table(top, widths, rows, row_height=40):
                xs = [30]
                for width in widths:
                    xs.append(xs[-1] + width)
                bottom = top + row_height * len(rows)
                for x in xs:
                    page.draw_line((x, top), (x, bottom))
                for index in range(len(rows) + 1):
                    y = top + row_height * index
                    page.draw_line((xs[0], y), (xs[-1], y))
                for row_index, row in enumerate(rows):
                    for column_index, cell in enumerate(row):
                        if cell:
                            page.insert_text(
                                (xs[column_index] + 5, top + row_index * row_height + 15),
                                cell, fontsize=10,
                            )

            page.insert_text((30, 30), "Before tables")
            draw_table(60, [100, 140, 160], [
                ["ID", "Value", "Note"],
                ["A-1", "pipe | slash \\", "line 1\nline 2"],
                ["B-2", "", "valid"],
            ])
            page.insert_text((445, 115), "Side note", fontsize=9)
            page.insert_text((30, 220), "Between tables")
            draw_table(250, [120, 120], [["Key", "Action"], ["C-3", "keep"]], row_height=30)
            page.insert_text((30, 350), "After tables")
            page = pdf.new_page(width=520, height=420)
            page.insert_text((30, 40), "No table page")
            pdf.save(path)

        raw = DocumentLoader().load_raw_file(str(path), loader_name="pdfplumber")
        cleaned = clean_documents(raw)
        before = "\n\n".join(doc.page_content for doc in raw)
        after = "\n\n".join(doc.page_content for doc in cleaned)
        _write_artifacts("PDF表格_特殊单元格和多页", path, before, after, cleaned)
        expected_table = "\n".join([
            "| ID | Value | Note |",
            "| --- | --- | --- |",
            r"| A-1 | pipe \| slash \\ | line 1<br>line 2 |",
        ])
        for text in (before, after):
            assert expected_table in text
            empty_row = next(line for line in text.splitlines() if line.startswith("| B-2 |"))
            assert [cell.strip() for cell in empty_row.split("|")[1:-1]] == ["B-2", "", "valid"]
            assert "| Key | Action |\n| --- | --- |\n| C-3 | keep |" in text
            assert "Side note" in text
            assert "No table page" in text
            assert text.index("Before tables") < text.index("| ID |") < text.index("Between tables")
            assert text.index("Between tables") < text.index("| Key |") < text.index("After tables")
            for identifier in ("A-1", "B-2", "C-3"):
                assert text.count(identifier) == 1
        assert len(cleaned) == 2
        assert [doc.metadata["page"] for doc in cleaned] == [1, 2]
        assert all(doc.metadata["loader"] == "PDFPlumberLoader" for doc in cleaned)
        assert raw[1].page_content == PDFPlumberLoader(str(path)).load()[1].page_content

    def test_pdf扫描件(self, monkeypatch):
        if not settings.TENCENT_OCR_SECRET_ID or not settings.TENCENT_OCR_SECRET_KEY:
            pytest.skip("请在本地 .env 配置腾讯 OCR 密钥后运行扫描件测试")
        monkeypatch.setattr(settings, "OCR_ENABLED", True)
        monkeypatch.setattr(settings, "OCR_PROVIDER", "tencent")
        path, _, cleaned, before, after = _load("扫描件测试.pdf", loader_name="pymupdf")
        _write_artifacts("PDF扫描件", path, before, after, cleaned)
        assert "扫描件测试" in after
        assert any(doc.metadata.get("ocr_used") for doc in cleaned)
        assert cleaned[0].metadata["loader"] == "PyMuPDFLoader"
