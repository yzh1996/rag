#!/usr/bin/env python
"""
Streamlit前端启动脚本
"""
import subprocess
import sys
from pathlib import Path
from config.settings import settings

if __name__ == "__main__":
    frontend_path = Path(__file__).parent / "frontend" / "app.py"
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(frontend_path),
        "--server.port", str(settings.STREAMLIT_PORT),
        "--server.address", settings.API_HOST
    ])