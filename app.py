import asyncio
import json
import os
import re
import time
from typing import Optional, Tuple

import numpy as np
from deep_translator import GoogleTranslator
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from faster_whisper import WhisperModel
from openai import AsyncOpenAI
from pydantic import BaseModel

try:
    import webrtcvad  # type: ignore
except Exception:
    webrtcvad = None

# ---------- Configuration ----------
DEFAULT_SOURCE_LANG = "ja"
DEFAULT_TARGET_LANG = "en"
NO_TRANSLATE_LANG = "none"

ASR_MODEL_SIZE = "small"
ASR_DEVICE = "cpu"  # set to "cuda" if GPU is available
ASR_COMPUTE_TYPE = "int8" if ASR_DEVICE == "cpu" else "float16"

PCM_SAMPLE_RATE = 16000
CHUNK_SEC = 2.0
VAD_RMS_THRESHOLD = 0.0008  # simple energy gate, tuned for int16 PCM
FINALIZE_SILENCE_SEC = 1.2
MAX_BUFFER_CHARS = 120
RING_TAIL_SEC = 0.5

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = "google/gemma-3-27b-it:free"
openrouter_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load Whisper once at startup
asr_model = WhisperModel(ASR_MODEL_SIZE, device=ASR_DEVICE, compute_type=ASR_COMPUTE_TYPE)


# ---------- Helpers ----------
def translate_google(text: str, source: str, target: str) -> str:
    text = text.strip()
    if not text:
        return ""
    translator = GoogleTranslator(source=source, target=target)
    return translator.translate(text)


def make_subtitle_json(
    original: str,
    translated: str,
    source_lang: str,
    target_lang: str,
    latency_ms: float,
    chunk_id: int,
    is_final: bool,
) -> str:
    now = time.strftime("%H:%M:%S")
    return json.dumps(
        {
            "original": original,
            "translated": translated,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "time": now,
            "latency_ms": round(latency_ms, 1),
            "chunk_id": chunk_id,
            "is_final": is_final,
        },
        ensure_ascii=False,
    )


class SummarizeRequest(BaseModel):
    history: list[dict]


# ---------- HTTP ----------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/summarize")
async def summarize(req: SummarizeRequest):
    if not req.history:
        return JSONResponse({"error": "history is empty"}, status_code=400)

    lines = []
    for item in req.history:
        original = item.get("original", item.get("jp", ""))
        translated = item.get("translated", item.get("en", ""))
        time_str = item.get("time", "")
        source = item.get("source_lang", "?")
        target = item.get("target_lang", "?")
        lines.append(f"[{time_str}] ({source}->{target}) Original: {original} / Translated: {translated}")

    history_text = "\n".join(lines)

    prompt = f"""
以下の対話ログを読み、主要トピックと結論を要約してください。
短い箇条書きで、重要なアクションがあれば明示してください。
## Summary (English)
- Main topics
- Key points
- Conclusions or action items (if any)
---
Conversation:
{history_text}
"""

    try:
        response = await openrouter_client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            extra_headers={
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "Whisper Subtitle App",
            },
        )

        if not response.choices:
            return JSONResponse({"error": f"API returned no choices: {response}"}, status_code=500)

        summary = response.choices[0].message.content
        if summary is None:
            return JSONResponse({"error": "summary was None"}, status_code=500)

        return JSONResponse({"summary": summary})
    except Exception as e:  # pragma: no cover - defensive
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------- WebSocket ----------
def run_whisper(audio: np.ndarray, language: Optional[str]) -> Tuple[str, object]:
    segments, info = asr_model.transcribe(
        audio,
        language=language,
        vad_filter=True,
        beam_size=3,
        condition_on_previous_text=False,
    )
    text = "".join([s.text for s in segments]).strip()
    return text, info


PUNCT_END_RE = re.compile(r"[。．\.!?！？]\s*$")


def merge_text(buffer: str, new_text: str) -> str:
    if not new_text:
        return buffer
    if not buffer:
        return new_text
    if new_text.startswith(buffer):
        return new_text
    if buffer.endswith(new_text):
        return buffer

    max_overlap = 0
    max_len = min(len(buffer), len(new_text))
    for i in range(1, max_len + 1):
        if buffer[-i:] == new_text[:i]:
            max_overlap = i
    if max_overlap > 0:
        return buffer + new_text[max_overlap:]
    return f"{buffer} {new_text}".strip()


def should_finalize(text: str) -> bool:
    if not text:
        return False
    if len(text) >= MAX_BUFFER_CHARS:
        return True
    return bool(PUNCT_END_RE.search(text))


async def process_audio_queue(queue: asyncio.Queue, websocket: WebSocket) -> None:
    print("[DEBUG] worker started")
    text_buffer = ""
    last_interim = ""
    last_voice_ts = 0.0
    last_detected_lang = DEFAULT_SOURCE_LANG
    last_target_lang = DEFAULT_TARGET_LANG
    tail_samples = int(PCM_SAMPLE_RATE * RING_TAIL_SEC)
    audio_tail = np.zeros(0, dtype=np.float32)
    vad = webrtcvad.Vad(2) if webrtcvad else None

    async def send_result(
        original_text: str,
        translated_text: str,
        source_lang: str,
        target_lang: str,
        latency_ms: float,
        chunk_id: int,
        is_final: bool,
    ) -> bool:
        payload = make_subtitle_json(
            original_text,
            translated_text,
            source_lang,
            target_lang,
            latency_ms,
            chunk_id,
            is_final,
        )
        try:
            await websocket.send_text(payload)
            kind = "final" if is_final else "interim"
            print(f"[DEBUG] chunk#{chunk_id} sent {kind} (latency={latency_ms:.1f}ms)")
            return True
        except Exception as e:
            print(f"[DEBUG] send failed: {e}")
            return False

    def webrtc_vad_has_voice(audio_f32: np.ndarray) -> bool:
        if vad is None:
            return False
        pcm16 = (audio_f32 * 32767.0).astype(np.int16).tobytes()
        frame_len = int(PCM_SAMPLE_RATE * 0.02) * 2  # 20ms * 2 bytes
        for i in range(0, len(pcm16) - frame_len + 1, frame_len):
            frame = pcm16[i : i + frame_len]
            if vad.is_speech(frame, PCM_SAMPLE_RATE):
                return True
        return False

    while True:
        try:
            item = await asyncio.wait_for(queue.get(), timeout=0.3)
        except asyncio.TimeoutError:
            if text_buffer and last_voice_ts > 0 and (time.perf_counter() - last_voice_ts) >= FINALIZE_SILENCE_SEC:
                translated = ""
                if last_target_lang != NO_TRANSLATE_LANG:
                    translated = await asyncio.to_thread(
                        translate_google, text_buffer, last_detected_lang, last_target_lang
                    )
                await send_result(
                    text_buffer,
                    translated,
                    last_detected_lang,
                    last_target_lang,
                    0.0,
                    0,
                    True,
                )
                text_buffer = ""
                last_interim = ""
            continue

        if item is None:
            return

        try:
            chunk_id, pcm_bytes, source_lang, target_lang = item
            last_target_lang = target_lang
            start = time.perf_counter()
            print(f"[DEBUG] worker got chunk#{chunk_id} ({len(pcm_bytes)} bytes)")

            audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            if audio.size == 0:
                print(f"[DEBUG] chunk#{chunk_id} empty audio buffer")
                continue

            rms = float(np.sqrt(np.mean(np.square(audio))))
            print(f"[DEBUG] chunk#{chunk_id} rms={rms:.5f}")

            voice_detected = rms >= VAD_RMS_THRESHOLD
            if vad:
                voice_detected = webrtc_vad_has_voice(audio)
                print(f"[DEBUG] chunk#{chunk_id} webrtcvad={voice_detected}")

            if not voice_detected:
                if text_buffer and last_voice_ts > 0 and (time.perf_counter() - last_voice_ts) >= FINALIZE_SILENCE_SEC:
                    translated = ""
                    if last_target_lang != NO_TRANSLATE_LANG:
                        translated = await asyncio.to_thread(
                            translate_google, text_buffer, last_detected_lang, last_target_lang
                        )
                    await send_result(
                        text_buffer,
                        translated,
                        last_detected_lang,
                        last_target_lang,
                        0.0,
                        chunk_id,
                        True,
                    )
                    text_buffer = ""
                    last_interim = ""
                continue

            last_voice_ts = time.perf_counter()

            if audio_tail.size > 0:
                audio_input = np.concatenate([audio_tail, audio])
            else:
                audio_input = audio

            if audio.size >= tail_samples:
                audio_tail = audio[-tail_samples:]
            else:
                audio_tail = audio

            whisper_lang = None if source_lang == "auto" else source_lang
            original, info = await asyncio.to_thread(run_whisper, audio_input, whisper_lang)

            detected_lang = source_lang
            if source_lang == "auto" and getattr(info, "language", None):
                detected_lang = info.language
            last_detected_lang = detected_lang

            if not original:
                print(f"[DEBUG] chunk#{chunk_id} empty result")
                continue

            text_buffer = merge_text(text_buffer, original)

            if text_buffer != last_interim:
                latency_ms = (time.perf_counter() - start) * 1000
                ok = await send_result(
                    text_buffer,
                    "",
                    detected_lang,
                    target_lang,
                    latency_ms,
                    chunk_id,
                    False,
                )
                if not ok:
                    return
                last_interim = text_buffer

            if should_finalize(text_buffer):
                translated = ""
                if target_lang != NO_TRANSLATE_LANG:
                    translated = await asyncio.to_thread(
                        translate_google, text_buffer, detected_lang, target_lang
                    )
                latency_ms = (time.perf_counter() - start) * 1000
                ok = await send_result(
                    text_buffer,
                    translated,
                    detected_lang,
                    target_lang,
                    latency_ms,
                    chunk_id,
                    True,
                )
                if not ok:
                    return
                text_buffer = ""
                last_interim = ""
        except Exception as e:
            print(f"[DEBUG] worker error on chunk#{item[0] if item else '?'}: {e}")


@app.websocket("/ws/audio")
async def ws_audio(websocket: WebSocket):
    await websocket.accept()
    print("[DEBUG] WebSocket connected")

    source_lang = DEFAULT_SOURCE_LANG
    target_lang = DEFAULT_TARGET_LANG
    queue: asyncio.Queue = asyncio.Queue()
    worker = asyncio.create_task(process_audio_queue(queue, websocket))
    print("[DEBUG] worker task created")
    chunk_id = 0

    try:
        while True:
            message = await websocket.receive()

            if "text" in message and message["text"] is not None:
                text_msg = message["text"]
                try:
                    data = json.loads(text_msg)
                    if data.get("type") == "config":
                        source_lang = data.get("source_lang", DEFAULT_SOURCE_LANG)
                        target_lang = data.get("target_lang", DEFAULT_TARGET_LANG)
                        print(f"[DEBUG] Config updated: {source_lang}->{target_lang}")
                        continue
                except json.JSONDecodeError:
                    # fallback: ignore non-JSON text
                    pass

                # Backward compatibility: base64 payloads (if any)
                try:
                    import base64

                    pcm_bytes = base64.b64decode(text_msg)
                    chunk_id += 1
                    await queue.put((chunk_id, pcm_bytes, source_lang, target_lang))
                    print(f"[DEBUG] queued base64 chunk#{chunk_id} ({len(pcm_bytes)} bytes)")
                except Exception:
                    print("[DEBUG] unexpected text message, skipped")
                continue

            if "bytes" in message and message["bytes"] is not None:
                pcm_bytes = message["bytes"]
                chunk_id += 1
                await queue.put((chunk_id, pcm_bytes, source_lang, target_lang))
                print(f"[DEBUG] queued binary chunk#{chunk_id} ({len(pcm_bytes)} bytes)")
                continue

    except WebSocketDisconnect:
        print("[DEBUG] WebSocket disconnected")
    finally:
        await queue.put(None)
        await asyncio.sleep(0)  # let worker exit
        if not worker.done():
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass


# ---------- Entry ----------
# uvicorn app:app --reload
