"""Fargate版の依存なし静的チェック（コンテナ実行前の簡易検証）。"""

import ast
import json
import re
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    base = Path(__file__).parent
    source = (base / "app.py").read_text(encoding="utf-8")
    html = (base / "templates" / "index.html").read_text(encoding="utf-8")
    dockerfile = (base / "Dockerfile").read_text(encoding="utf-8")

    # Python構文の検証（モデルのダウンロードやimportは実行しない）
    ast.parse(source, filename="app.py")
    json.loads((base / "ecs-task-definition.example.json").read_text(encoding="utf-8"))

    # 機密会議向けコンテナに外部翻訳・要約サービスが含まれないことを確認
    for forbidden in ("GoogleTranslator", "AsyncOpenAI", "openrouter.ai"):
        assert forbidden not in source, forbidden

    assert 'APP_ACCESS_TOKEN = os.getenv("APP_ACCESS_TOKEN", "")' in source
    assert "secrets.compare_digest" in source
    assert 'MAX_AUDIO_QUEUE_CHUNKS = 32' in source
    assert '@app.get("/healthz")' in source
    assert "cell.textContent = value" in html
    assert "new WebSocket" in html
    assert "download_model(" in dockerfile
    assert "COPY app.py bedrock_text.py streaming_asr.py transcript_stability.py quality_guard.py ." in dockerfile
    streaming = (base / "streaming_asr.py").read_text(encoding="utf-8")
    ast.parse(streaming, filename="streaming_asr.py")
    ast.parse((base / "test_streaming_asr.py").read_text(encoding="utf-8"), filename="test_streaming_asr.py")
    assert "UtteranceBuffer" in source
    assert "feed_many(audio)" in source
    assert "asr_inference_semaphore" in source
    assert "ASR_INTERIM_BACKLOG_THRESHOLD" in source
    assert "session_id = secrets.token_hex(8)" in source
    assert "utterance_id: str" in source
    assert "item.utterance_id === data.utterance_id" in html
    assert "def feed_many(" in streaming
    assert "vad_parameters=" in source
    assert "ASR_VOCAB_PROMPT" in source
    assert '"type": "finished"' in source
    assert "new Int16Array(pcmBuffer).buffer" in html
    assert "VAD_THRESHOLD" not in html
    assert "CHUNK_SEC = 0.5" in html
    assert "ASR_HOTWORDS" in source
    assert "ASR_END_SILENCE_SEC" in source
    assert "ASR_PREROLL_SEC" in source
    assert "self._preroll" in streaming
    for path in ("accuracy_metrics.py", "benchmark_asr.py", "test_accuracy_metrics.py",
                 "transcript_stability.py", "quality_guard.py", "test_vexa_quality.py",
                 "replay_asr.py", "test_replay_asr.py"):
        ast.parse((base / path).read_text(encoding="utf-8"), filename=path)
    assert '"type": "translation"' in source
    assert "ASR_CONTEXT_RESET_SEC" in source
    assert "audio_start_ms" in source and "audio_end_ms" in source
    assert "LocalAgreement" in source
    assert "reject_low_confidence_segment" in source
    assert "stable_prefix" in html
    assert "utterance_id" in html
    assert 'BEDROCK_MODEL_ID' in source
    assert 'translate_bedrock' in source
    assert 'bedrock:InvokeModel' in (base / "README.md").read_text(encoding="utf-8")
    bedrock = (base / "bedrock_text.py").read_text(encoding="utf-8")
    ast.parse(bedrock, filename="bedrock_text.py")
    assert 'bedrock-runtime' in bedrock
    assert '.converse(' in bedrock
    assert 'Authorization' in html

    # Node.jsがある場合はフロントエンドのJavaScript構文も検証
    match = re.search(r"<script>(.*?)</script>", html, flags=re.DOTALL)
    assert match is not None
    if shutil.which("node"):
        subprocess.run(["node", "--check"], input=match.group(1), text=True, encoding="utf-8", check=True)
    print("static checks passed")


if __name__ == "__main__":
    main()
