"""上传大小、实际格式与Office解压上限检查。"""
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from config.settings import settings

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md", ".xlsx", ".xls"}


def validate_file(path):
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError("不支持的文件格式")
    if not path.stat().st_size:
        raise ValueError("不能上传空文件")
    if path.stat().st_size > settings.MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"文件超过{settings.MAX_UPLOAD_MB}MB上限")
    with path.open("rb") as stream:
        header = stream.read(1024)
    if ext == ".pdf" and not header.lstrip().startswith(b"%PDF-"):
        raise ValueError("文件内容不是有效的PDF")
    if ext in {".doc", ".xls"} and not header.startswith(bytes.fromhex("D0CF11E0A1B11AE1")):
        raise ValueError("文件内容与Office格式不符")
    if ext in {".docx", ".xlsx"}:
        try:
            with ZipFile(path) as archive:
                entries = archive.infolist()
                if len(entries) > settings.MAX_ARCHIVE_ENTRIES or sum(i.file_size for i in entries) > settings.MAX_ARCHIVE_EXPANDED_MB * 1024 * 1024:
                    raise ValueError("Office文件解压后过大，拒绝处理")
                if any(i.flag_bits & 1 or ".." in Path(i.filename.replace("\\", "/")).parts or i.filename.startswith(("/", "\\")) or ":" in i.filename for i in entries):
                    raise ValueError("Office文件包含不安全或加密内容")
                required = "word/document.xml" if ext == ".docx" else "xl/workbook.xml"
                if required not in archive.namelist():
                    raise ValueError("文件内容与Office格式不符")
        except BadZipFile as exc:
            raise ValueError("Office文件损坏或格式不符") from exc
    if ext in {".txt", ".md"}:
        try:
            with path.open(encoding="utf-8-sig") as stream:
                while stream.read(1024 * 1024):
                    pass
        except UnicodeError as exc:
            raise ValueError("文本文件请使用UTF-8编码") from exc


def save_upload(upload, destination):
    """分段读取，超限立即停止；只清理本次上传创建的文件。"""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    created = False
    try:
        with destination.open("xb") as stream:
            created = True
            while True:
                chunk = upload.file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > settings.MAX_UPLOAD_MB * 1024 * 1024:
                    raise ValueError(f"文件超过{settings.MAX_UPLOAD_MB}MB上限")
                stream.write(chunk)
        validate_file(destination)
    except Exception:
        if created:
            destination.unlink(missing_ok=True)
        raise
