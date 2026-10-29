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
    assert 'MAX_AUDIO_QUEUE_CHUNKS = 8' in source
    assert '@app.get("/healthz")' in source
    assert "cell.textContent = value" in html
    assert "new WebSocket" in html
    assert "download_model(" in dockerfile
    assert "COPY app.py bedrock_text.py ." in dockerfile
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
        subprocess.run(["node", "--check"], input=match.group(1), text=True, check=True)
    print("static checks passed")


if __name__ == "__main__":
    main()
