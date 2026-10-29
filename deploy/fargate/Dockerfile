FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# CTranslate2 (faster-whisper) のCPU実行に必要なOpenMPランタイム
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

# コンテナ作成時のみHugging Faceからモデルを取得。実行時の外部DLは不要。
ARG MODEL_SIZE=small
RUN python -c "from faster_whisper.utils import download_model; download_model('${MODEL_SIZE}', output_dir='/opt/models/${MODEL_SIZE}')"
ENV ASR_MODEL_PATH=/opt/models/${MODEL_SIZE} \
    ASR_CPU_THREADS=4

COPY app.py bedrock_text.py streaming_asr.py transcript_stability.py quality_guard.py .
COPY templates/ ./templates/

# コンテナを非rootで実行
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin appuser
USER appuser

EXPOSE 8000

# 音声認識モデルを重複ロードしないようworkerは1つ
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--ws-max-size", "131072"]
