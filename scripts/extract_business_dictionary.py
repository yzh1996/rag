"""Extract the checked-in field dictionary without inventing missing definitions."""
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile
import hashlib
import json
import re


ROOT = Path(__file__).resolve().parents[1]


def extract_dictionary(source):
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with ZipFile(source) as archive:
        xml = archive.read("word/document.xml")
    body = ET.fromstring(xml).find("w:body", ns)
    def text(node):
        return "".join(item.text or "" for item in node.findall(".//w:t", ns)).strip()
    tables, current, layer = [], None, ""
    for node in body:
        if node.tag.endswith("}p"):
            value = text(node)
            if re.match(r"3\.\d+ ", value):
                layer = value.split(" ", 1)[1]
            match = re.match(r"^([a-zA-Z_][a-zA-Z_0-9]*)\s*[（(]", value)
            if match:
                current = match.group(1)
        elif node.tag.endswith("}tbl") and current:
            rows = [[text(cell) for cell in row.findall("w:tc", ns)]
                    for row in node.findall("w:tr", ns)]
            columns = []
            for row in rows[1:]:
                if len(row) < 6 or row[0] == "源文档未提取到字段":
                    continue
                columns.append(dict(zip(("name", "description", "source_type", "nullable", "notes", "source"), row)))
            tables.append({"name": current, "layer": layer,
                           "status": "defined" if columns else "missing_columns", "columns": columns})
            current = None
    return {"source": source.name, "source_xml_sha256": hashlib.sha256(xml).hexdigest(),
            "tables": tables}


if __name__ == "__main__":
    source = ROOT / "doc" / "六个维度大数据平台数据表与字段字典.docx"
    catalog = extract_dictionary(source)
    target = ROOT / "config" / "business_schema.json"
    target.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"tables": len(catalog["tables"]),
                      "missing_columns": [t["name"] for t in catalog["tables"] if not t["columns"]]}, ensure_ascii=False))
