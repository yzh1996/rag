"""Checksummed offline backups and restore to an empty recovery environment."""
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
import zipfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url

from config.settings import settings


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def postgres_command(executable, url, *arguments):
    connection = make_url(url)
    env = {**os.environ, "PGPASSWORD": connection.password or ""}
    for name in ("sslmode", "sslrootcert", "sslcert", "sslkey"):
        if name in connection.query:
            env["PG" + name.upper()] = connection.query[name]
    command = [executable, "--host", connection.host or "localhost", "--port", str(connection.port or 5432),
        "--username", connection.username or "postgres", "--dbname", connection.database or "postgres", *arguments]
    subprocess.run(command, env=env, check=True, capture_output=True)


def s3_client():
    import boto3

    return boto3.client("s3", endpoint_url=settings.S3_ENDPOINT_URL,
        aws_access_key_id=settings.S3_ACCESS_KEY, aws_secret_access_key=settings.S3_SECRET_KEY,
        region_name=settings.S3_REGION)


def _create_backup(output: Path, offline_confirmed: bool, upload=False):
    if not offline_confirmed:
        raise ValueError("Stop API and ingestion writers before backup; explicitly confirm offline mode")
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Backup output already exists")
    for directory in (settings.UPLOAD_DIR, settings.INDEX_SNAPSHOT_DIR, settings.VECTOR_DB_DIR):
        if directory.is_symlink() or any(path.is_symlink() for path in directory.rglob("*")):
            raise ValueError("Backup source must not contain symbolic links")
    with tempfile.TemporaryDirectory(prefix="rag-backup-") as temporary:
        root = Path(temporary)
        driver = make_url(settings.DATABASE_URL).get_backend_name()
        if driver == "sqlite":
            source = Path(make_url(settings.DATABASE_URL).database).resolve()
            with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as db:
                with closing(sqlite3.connect(root / "database.sqlite")) as target:
                    db.backup(target)
        elif driver == "postgresql":
            postgres_command("pg_dump", settings.DATABASE_URL, "--format=custom", "--no-owner", "--file", str(root / "database.dump"))
        else:
            raise ValueError("Unsupported backup database")
        if settings.UPLOAD_DIR.exists():
            shutil.copytree(settings.UPLOAD_DIR, root / "uploads")
        if settings.INDEX_SNAPSHOT_DIR.exists():
            shutil.copytree(settings.INDEX_SNAPSHOT_DIR, root / "index_snapshots")
        if settings.VECTOR_STORE_TYPE == "opensearch":
            from opensearchpy.helpers import scan
            from core.opensearch_store import OpenSearchStore

            store = OpenSearchStore(None, initialize=False)
            if not store.client.indices.exists(index=store.index):
                raise ValueError("Backup OpenSearch index does not exist")
            (root / "search_mapping.json").write_text(json.dumps(store.client.indices.get_mapping(index=store.index)), encoding="utf-8")
            with (root / "search.ndjson").open("w", encoding="utf-8") as stream:
                for hit in scan(store.client, index=store.index, query={"query": {"match_all": {}}}):
                    stream.write(json.dumps({"id": hit["_id"], "source": hit["_source"]}, ensure_ascii=True) + "\n")
        elif settings.VECTOR_DB_DIR.exists():
            shutil.copytree(settings.VECTOR_DB_DIR, root / "vectors")
        files = {path.relative_to(root).as_posix(): digest(path) for path in root.rglob("*") if path.is_file()}
        manifest = {"format": 1, "created_at": datetime.now(timezone.utc).isoformat(), "database": driver,
            "search": settings.VECTOR_STORE_TYPE, "dimension": settings.OPENSEARCH_DIMENSION,
            "embedding_model": settings.EMBEDDING_MODEL_NAME, "files": files,
            "upload_root": str(settings.UPLOAD_DIR.resolve()), "snapshot_root": str(settings.INDEX_SNAPSHOT_DIR.resolve())}
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in root.rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(root).as_posix())
    if upload:
        client = s3_client()
        try:
            client.head_bucket(Bucket=settings.S3_BUCKET)
        except client.exceptions.ClientError as exc:
            if str(exc.response["Error"]["Code"]) not in {"404", "NoSuchBucket"}:
                raise
            client.create_bucket(Bucket=settings.S3_BUCKET)
        client.put_bucket_versioning(Bucket=settings.S3_BUCKET, VersioningConfiguration={"Status": "Enabled"})
        key = "backups/" + output.name
        client.upload_file(str(output), settings.S3_BUCKET, key, ExtraArgs={"Metadata": {"sha256": digest(output)}})
        return {"path": str(output), "bucket": settings.S3_BUCKET, "key": key, "sha256": digest(output)}
    return {"path": str(output), "sha256": digest(output)}


def create_backup(output: Path, offline_confirmed: bool, upload=False):
    status_path = settings.DATA_DIR / "backup_status.json"
    try:
        result = _create_backup(output, offline_confirmed, upload)
    except Exception:
        if offline_confirmed:
            status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
            status["last_attempt_failed"] = True
            status_path.parent.mkdir(parents=True, exist_ok=True)
            status_path.write_text(json.dumps(status), encoding="utf-8")
        raise
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps({"last_success_timestamp": datetime.now(timezone.utc).timestamp(),
        "last_attempt_failed": False}), encoding="utf-8")
    return result


def verify_backup(archive_path, destination):
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate backup members")
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in info.filename or ":" in info.filename or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Unsafe backup path")
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("format") != 1 or set(names) != set(manifest["files"]) | {"manifest.json"}:
            raise ValueError("Invalid backup manifest")
        for name, expected in manifest["files"].items():
            target = Path(destination) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
            if digest(target) != expected:
                raise ValueError("Backup checksum mismatch: " + name)
    return manifest


def restore_backup(archive_path, offline_confirmed: bool, execute=False):
    if not offline_confirmed:
        raise ValueError("Stop recovery API before restore and confirm offline mode")
    with tempfile.TemporaryDirectory(prefix="rag-restore-") as temporary:
        root = Path(temporary)
        manifest = verify_backup(archive_path, root)
        driver = make_url(settings.DATABASE_URL).get_backend_name()
        if manifest["database"] != driver or manifest["search"] != settings.VECTOR_STORE_TYPE:
            raise ValueError("Recovery database/search must match backup")
        if manifest["embedding_model"] != settings.EMBEDDING_MODEL_NAME or manifest["dimension"] != settings.OPENSEARCH_DIMENSION:
            raise ValueError("Recovery embedding configuration differs from backup")
        if str(settings.UPLOAD_DIR.resolve()) != manifest["upload_root"] or str(settings.INDEX_SNAPSHOT_DIR.resolve()) != manifest["snapshot_root"]:
            raise ValueError("Use the same in-container data paths so stored file references remain valid")
        for path in (settings.UPLOAD_DIR, settings.INDEX_SNAPSHOT_DIR, settings.VECTOR_DB_DIR):
            if path.exists() and any(path.iterdir()):
                raise ValueError("Recovery file destination is not empty")
        engine = None
        if driver == "sqlite":
            database_path = Path(make_url(settings.DATABASE_URL).database)
            if database_path.exists():
                raise ValueError("Recovery SQLite file already exists")
        else:
            engine = create_engine(settings.DATABASE_URL)
            if inspect(engine).get_table_names():
                engine.dispose()
                raise ValueError("Recovery PostgreSQL database is not empty")
        store = None
        if settings.VECTOR_STORE_TYPE == "opensearch":
            from core.opensearch_store import OpenSearchStore

            store = OpenSearchStore(None, initialize=False)
            if store.client.indices.exists(index=store.index) and store.count():
                raise ValueError("Recovery OpenSearch index is not empty")
        if not execute:
            if engine:
                engine.dispose()
            return {"validated": True, "execute": False, "created_at": manifest["created_at"]}
        if driver == "sqlite":
            database_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / "database.sqlite", database_path)
        else:
            postgres_command("pg_restore", settings.DATABASE_URL, "--no-owner", "--no-privileges", "--single-transaction", str(root / "database.dump"))
            engine.dispose()
        for source, target in (("uploads", settings.UPLOAD_DIR), ("index_snapshots", settings.INDEX_SNAPSHOT_DIR), ("vectors", settings.VECTOR_DB_DIR)):
            if (root / source).exists():
                shutil.copytree(root / source, target, dirs_exist_ok=True)
        if store:
            from opensearchpy.helpers import bulk

            store._ensure_index()
            def actions():
                with (root / "search.ndjson").open(encoding="utf-8") as stream:
                    for line in stream:
                        item = json.loads(line)
                        yield {"_index": store.index, "_id": item["id"], "_source": item["source"]}
            bulk(store.client, actions(), refresh="wait_for", raise_on_error=True)
        return {"validated": True, "execute": True, "created_at": manifest["created_at"]}
