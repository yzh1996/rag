"""Offline backup, S3 retrieval, verification and empty-environment recovery."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.backup_service import create_backup, digest, restore_backup, s3_client
from config.settings import settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["create", "restore", "download"])
    parser.add_argument("path", type=Path)
    parser.add_argument("--confirm-offline", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--key")
    options = parser.parse_args()
    if options.action == "create":
        result = create_backup(options.path, options.confirm_offline, options.upload)
    elif options.action == "restore":
        result = restore_backup(options.path, options.confirm_offline, options.execute)
    else:
        if not options.key or options.path.exists():
            parser.error("Download requires --key and a new output path")
        client = s3_client()
        metadata = client.head_object(Bucket=settings.S3_BUCKET, Key=options.key)
        client.download_file(settings.S3_BUCKET, options.key, str(options.path))
        if digest(options.path) != metadata.get("Metadata", {}).get("sha256"):
            raise ValueError("Downloaded backup checksum mismatch")
        result = {"path": str(options.path), "verified": True}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
