"""Export tool descriptions generated from the executable tool registry."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.assistant_tool_catalog import render_catalog_markdown
if __name__ == "__main__":
    destination = Path(__file__).resolve().parents[1] / "doc" / "assistant_tool_catalog.md"
    destination.write_text(render_catalog_markdown(), encoding="utf-8")
    print(destination)
