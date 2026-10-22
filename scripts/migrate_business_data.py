"""Preview/import existing SQLite records into a new PostgreSQL database."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import settings
from core.business_migration import import_sqlite


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--source-upload-root")
    parser.add_argument("--execute", action="store_true")
    options = parser.parse_args()
    print(json.dumps(import_sqlite(options.source, settings.DATABASE_URL,
        options.source_upload_root, str(settings.UPLOAD_DIR), options.execute)))


if __name__ == "__main__":
    main()
