# Whisper Subtitle App

Real-time speech-to-text with optional translation and AI summaries.

## Requirements

- Python 3.9+
- ffmpeg **not required** (audio is sent as 16kHz PCM from the browser)
- Optional: `webrtcvad` for stronger VAD

## Install

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

Optional VAD (recommended):

```bash
pip install webrtcvad
```

## .env

Create `.env` in the project root:

```
OPENROUTER_API_KEY=sk-or-v1-...
```

## Run

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open in a browser:

```
http://localhost:8000/
```

## Notes

- System audio capture may require HTTPS depending on your browser.
- Translation can be disabled in the UI (transcription-only mode).
- Summary prompt is editable in the UI and sent with the summarize request.

## License

This application's original source code is licensed under the **MIT License**.
See [LICENSE](LICENSE). Commercial use, modification, and redistribution are
permitted provided the copyright and license notice is retained.

Third-party libraries and model weights retain their own licenses. See the
[Fargate third-party notices](deploy/fargate/THIRD_PARTY_NOTICES.md).
If you redistribute a container image, check the exact installed package
licenses and include notices required by those components.

**Note:** A permissive license does not grant permission to record other
participants without required consent or bypass applicable terms of service.
