"""
transcriber.py
Transcribes an audio file using Faster-Whisper.
Returns structured JSON with word-level confidence and timestamps.
"""

import json
import time
import logging
from pathlib import Path
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def transcribe(audio_path: str, model_size: str = "base") -> dict:
    """
    Transcribe audio and return structured result.
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    logger.info(f"Loading Faster-Whisper model: {model_size}")

    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    start_time = time.perf_counter()

    logger.info(f"Transcribing: {audio_path.name}")

    segments, info = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        language="en",
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    latency = time.perf_counter() - start_time

    all_words = []
    all_segments = []
    full_text_parts = []

    for seg in segments:
        seg_data = {
            "id": seg.id,
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
            "avg_logprob": round(seg.avg_logprob, 4),
            "no_speech_prob": round(seg.no_speech_prob, 4),
        }

        all_segments.append(seg_data)
        full_text_parts.append(seg.text.strip())

        if seg.words:
            for w in seg.words:
                all_words.append({
                    "word": w.word.strip(),
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                    "confidence": round(w.probability, 4),
                })

    result = {
        "transcript": " ".join(full_text_parts),
        "language": info.language,
        "language_probability": round(info.language_probability, 4),
        "duration_seconds": round(info.duration, 2),
        "transcription_latency_seconds": round(latency, 3),
        "model_used": f"faster-whisper-{model_size}",
        "words": all_words,
        "segments": all_segments,
    }

    if not result["transcript"].strip():
        logger.warning("Transcript is empty — audio may be silent or corrupted.")
        result["warning"] = "empty_transcript"

    return result


def save_transcript(result: dict, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(f"Transcript saved to: {output_path}")


if __name__ == "__main__":
    result = transcribe("input/investor_sample.mp3")
    save_transcript(result, "output/transcript.json")

    print(f"Transcript: {result['transcript']}")
    print(f"Latency: {result['transcription_latency_seconds']}s")