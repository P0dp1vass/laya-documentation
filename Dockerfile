FROM python:3.12-slim
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 USE_TF=0 USE_TORCH=1 TOKENIZERS_PARALLELISM=false HF_HOME=/home/app/.cache/huggingface
WORKDIR /app
COPY requirements.txt constraints-reference.txt ./
RUN python -m pip install --no-cache-dir torch==2.8.0 --index-url ${TORCH_INDEX_URL} \
    && python -m pip install --no-cache-dir -r requirements.txt -c constraints-reference.txt \
    && useradd --create-home --uid 10001 app \
    && install -d -o app -g app /home/app/.cache/huggingface /app/outputs
COPY --chown=app:app . .
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=300s --retries=3 CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=2)"]
CMD ["uvicorn", "production.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--limit-concurrency", "32"]
