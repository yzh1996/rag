FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 postgresql-client tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-eng && rm -rf /var/lib/apt/lists/*
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY api api
COPY core core
COPY config config
COPY frontend frontend
COPY scripts scripts
COPY run_worker.py run_worker.py
RUN useradd --create-home --uid 10001 rag && mkdir -p /app/data /app/upload /app/vector_db /app/models && chown -R rag:rag /app
USER rag
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
