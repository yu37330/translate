"""Evaluate CPU faster-whisper on LOCAL 16-kHz mono PCM WAV clips.

Never uploads audio or transcripts to an external service.
Usage:
 python benchmark_asr.py --manifest manifest.jsonl --model-path /opt/models/small
Each JSONL row: {"id":"meeting-01","wav":"audio/clip01.wav","reference":"...","language":"ja"}
Optionally provide --terms-file containing one domain term per line.
"""
import argparse
import json
import os
from pathlib import Path
import time
import wave

import numpy as np

from accuracy_metrics import evaluate_pairs


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as f:
        if f.getnchannels() != 1 or f.getframerate() != 16000 or f.getsampwidth() != 2:
            raise ValueError(f"{path}: requires mono, 16 kHz, 16-bit PCM WAV")
        raw = f.readframes(f.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--beam-size", type=int, default=3)
    parser.add_argument("--vocab-prompt", default="")
    parser.add_argument("--hotwords", default="")
    parser.add_argument("--terms-file", type=Path)
    parser.add_argument("--output", type=Path, help="Optional JSON report (NO audio/transcript content)")
    args = parser.parse_args()
    if not args.model_path.is_dir():
        parser.error("model directory not found; this benchmark never downloads models")
    from faster_whisper import WhisperModel

    model = WhisperModel(str(args.model_path), device="cpu", compute_type="int8",
                         cpu_threads=args.cpu_threads, local_files_only=True)
    manifest_dir = args.manifest.resolve().parent
    cases = []
    for line in args.manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            case = json.loads(line)
            wav_path = (manifest_dir / case["wav"]).resolve()
            if not wav_path.is_relative_to(manifest_dir):
                raise ValueError("Audio file must remain under manifest directory")
            cases.append((case, wav_path))
    if not cases:
        parser.error("empty manifest")
    terms = (args.terms_file.read_text(encoding="utf-8").splitlines()
             if args.terms_file else [])

    pairs = []
    inference_seconds, audio_seconds = 0.0, 0.0
    for case, wav_path in cases:
        samples = read_wav(wav_path)
        start = time.perf_counter()
        segments, _ = model.transcribe(
            samples, language=case.get("language", "ja"),
            beam_size=args.beam_size,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 500,
                "min_speech_duration_ms": 100,
                "speech_pad_ms": 200,
            },
            initial_prompt=args.vocab_prompt or None,
            hotwords=args.hotwords or None,
            condition_on_previous_text=False,
        )
        hypothesis = "".join(segment.text for segment in segments).strip()
        inference_seconds += time.perf_counter() - start
        audio_seconds += len(samples) / 16000
        pairs.append({"id": case.get("id", wav_path.name),
                      "reference": case["reference"], "hypothesis": hypothesis})
    report = evaluate_pairs(pairs, terms)
    report["audio_seconds"] = round(audio_seconds, 2)
    report["inference_seconds"] = round(inference_seconds, 2)
    report["realtime_factor"] = round(inference_seconds / audio_seconds, 3)
    report["model_path"] = str(args.model_path)
    public_report = json.dumps(report, ensure_ascii=False, indent=2)
    print(public_report)
    if args.output:
        args.output.write_text(public_report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
