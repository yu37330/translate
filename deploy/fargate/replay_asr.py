"""Offline replay of the EXACT 500ms transport + utterance-buffer decisions.

This uses a pre-downloaded local CTranslate2 Whisper model (never fetches one).
It prints only aggregate CER, counts, and timing, not meeting transcripts.

  python replay_asr.py --manifest C:/eval/manifest.jsonl --model-path C:/models/small

Manifest format matches benchmark_asr.py. Use consented, de-identified audio
outside the repository. Keep transcript text out of Git and CloudWatch logs.
"""
import argparse
import json
from pathlib import Path
import time

from accuracy_metrics import evaluate_pairs
from benchmark_asr import read_wav
from quality_guard import reject_low_confidence_segment
from streaming_asr import UtteranceBuffer
from transcript_stability import LocalAgreement


def run_clip(samples, transcribe, *, sample_rate=16000, chunk_sec=0.5):
    """Inject input in exact audio-time frames. transcribe(audio)->str.

    A fake transcriber can be used in tests without Whisper or AWS.
    """
    buffer = UtteranceBuffer(sample_rate=sample_rate)
    agreement = LocalAgreement()
    final_parts = []
    interim_count = 0
    stable_count = 0
    utterance_windows = []
    previous_interim = ""
    step = int(sample_rate * chunk_sec)

    def publish(event):
        nonlocal interim_count, stable_count, previous_interim
        if event is None:
            return
        kind, audio = event
        text = transcribe(audio)
        if kind == "interim":
            interim_count += 1
            if agreement.observe(text):
                stable_count += 1
            previous_interim = text
            return
        final = text or previous_interim
        previous_interim = ""
        agreement.reset()
        if final:
            final_parts.append(final)
            utterance_windows.append((
                buffer.last_event_start_sample,
                buffer.last_event_end_sample,
            ))

    for start in range(0, len(samples), step):
        publish(buffer.feed(samples[start:start + step]))
    publish(("final", buffer.flush()) if buffer.active else None)
    return {
        "text": " ".join(final_parts).strip(),
        "utterances": len(final_parts),
        "interim_updates": interim_count,
        "stability_matches": stable_count,
        "windows": utterance_windows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--beam-size", type=int, default=3)
    parser.add_argument("--vocab-prompt", default="")
    parser.add_argument("--hotwords", default="")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.model_path.is_dir():
        parser.error("Model must already exist locally. No network download is supported.")
    from faster_whisper import WhisperModel

    model = WhisperModel(str(args.model_path), device="cpu", compute_type="int8",
                         cpu_threads=args.cpu_threads, local_files_only=True)
    base = args.manifest.resolve().parent

    def transcribe(audio):
        segments, _ = model.transcribe(
            audio, language="ja", beam_size=args.beam_size,
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
        return "".join(s.text for s in segments if not reject_low_confidence_segment(s)).strip()

    rows, n_calls = [], 0
    total_audio, total_inference = 0.0, 0.0
    total_utterances, total_interims, total_stables = 0, 0, 0

    def measured_transcribe(audio):
        nonlocal n_calls
        n_calls += 1
        return transcribe(audio)

    for line in args.manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        path = (base / case["wav"]).resolve()
        if not path.is_relative_to(base):
            raise ValueError("WAV must be under the manifest directory")
        samples = read_wav(path)
        began = time.perf_counter()
        result = run_clip(samples, measured_transcribe)
        total_inference += time.perf_counter() - began
        total_audio += len(samples) / 16000
        rows.append({"id": case.get("id", len(rows)+1), "reference": case["reference"],
                     "hypothesis": result["text"]})
        total_utterances += result["utterances"]
        total_interims += result["interim_updates"]
        total_stables += result["stability_matches"]
    if not rows:
        parser.error("Empty manifest")
    report = evaluate_pairs(rows)
    report.update({
        "audio_seconds": round(total_audio, 2),
        "replay_wall_seconds": round(total_inference, 2),
        "replay_runtime_factor": round(total_inference / total_audio, 3),
        "inference_calls": n_calls,
        "final_utterances": total_utterances,
        "interim_updates": total_interims,
        "stable_prefix_events": total_stables,
        "mode": "offline-replay; browser capture and realtime network latency NOT measured",
    })
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    print(payload)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
