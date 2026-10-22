"""Explicit read-only SQLite import into an empty PostgreSQL database."""
from pathlib import Path, PurePosixPath, PureWindowsPath

from sqlalchemy import DateTime, JSON, MetaData, Table, create_engine, inspect, select
from sqlalchemy.engine import make_url
import json
from datetime import datetime

from core.database import Base
from core import db_models  # noqa: F401


def remap_file(value, source_root, destination_root):
    if not value or not source_root:
        return value
    path_type = PureWindowsPath if PureWindowsPath(source_root).drive else PurePosixPath
    path = path_type(value)
    try:
        relative = path.relative_to(path_type(source_root))
    except ValueError as exc:
        raise ValueError("Stored document path lies outside the supplied upload root") from exc
    return str(PurePosixPath(destination_root).joinpath(*relative.parts))


def import_sqlite(source_path, destination_url, source_upload_root=None, destination_upload_root="/app/upload", execute=False):
    source_path = Path(source_path).resolve()
    if not source_path.is_file():
        raise ValueError("Source SQLite database does not exist")
    if make_url(destination_url).get_backend_name() != "postgresql":
        raise ValueError("Enterprise import target must be PostgreSQL")
    import sqlite3
    source = create_engine("sqlite://", creator=lambda: sqlite3.connect(source_path.as_uri() + "?mode=ro", uri=True))
    destination = create_engine(destination_url)
    try:
        if inspect(destination).get_table_names():
            raise ValueError("Import destination must be a new empty database")
        metadata = MetaData()
        source_names = set(inspect(source).get_table_names())
        datasets = []
        counts = {}
        with source.connect() as connection:
            for target in Base.metadata.sorted_tables:
                if target.name not in source_names:
                    continue
                old = Table(target.name, metadata, autoload_with=source)
                rows = []
                for record in connection.execute(select(old)).mappings():
                    row = {key: value for key, value in record.items() if key in target.c}
                    for key, value in list(row.items()):
                        if value is None and not target.c[key].nullable and target.c[key].default is not None:
                            row.pop(key)
                            continue
                        if isinstance(target.c[key].type, JSON) and isinstance(value, str):
                            row[key] = json.loads(value)
                        if isinstance(target.c[key].type, DateTime) and isinstance(value, str):
                            row[key] = datetime.fromisoformat(value)
                    for field in ("source_uri", "file_path"):
                        if field in row:
                            row[field] = remap_file(row[field], source_upload_root, destination_upload_root)
                    rows.append(row)
                datasets.append((target, rows))
                counts[target.name] = len(rows)
        if execute:
            with destination.begin() as connection:
                Base.metadata.create_all(connection)
                for table, rows in datasets:
                    for row in rows:
                        connection.execute(table.insert(), row)
        return {"execute": execute, "counts": counts}
    finally:
        source.dispose()
        destination.dispose()
