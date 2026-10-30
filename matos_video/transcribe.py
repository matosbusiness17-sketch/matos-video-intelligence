"""Audio extraction and Faster-Whisper transcription module."""
import json
import subprocess
import time
from pathlib import Path
from typing import Optional, List, Tuple, Any

from matos_video.config import (
    FFMPEG_PATH,
    MODELS_DIR,
    logger,
    TranscriptResult,
    TranscriptSegment,
    WordTimestamp,
)

def extract_audio(video_path: str, output_wav: str) -> str:
    """Extracts a 16kHz mono 16-bit PCM WAV audio file from a video using FFmpeg."""
    in_path = Path(video_path).resolve()
    out_path = Path(output_wav).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        FFMPEG_PATH,
        "-y",
        "-i", str(in_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(out_path),
    ]

    logger.info(f"Extracting 16kHz mono audio from '{in_path.name}' -> '{out_path.name}'")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg audio extraction failed (code {result.returncode}): {result.stderr.strip()}")

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"Extracted audio file is missing or empty: {out_path}")

    logger.info(f"Audio extracted successfully: {out_path.stat().st_size} bytes")
    return str(out_path)

def _probe_and_load_whisper_model(model_size: str = "tiny") -> Tuple[Any, str, str]:
    """Loads WhisperModel with live CUDA forward verification and seamless CPU fallback."""
    from faster_whisper import WhisperModel
    import ctranslate2
    import numpy as np

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Probe CUDA with live execution test
    can_try_cuda = False
    try:
        if ctranslate2.get_cuda_device_count() > 0:
            can_try_cuda = True
    except Exception:
        can_try_cuda = False

    if can_try_cuda:
        try:
            logger.info(f"Attempting to load WhisperModel('{model_size}') on CUDA (float16)...")
            model = WhisperModel(model_size, device="cuda", compute_type="float16", download_root=str(MODELS_DIR))
            # Test forward pass with 0.1s dummy features to verify cublas64_12.dll
            dummy_features = np.zeros((1, 80, 100), dtype=np.float32)
            # Use ctranslate2 StorageView to probe cublas
            _ = model.model.encode(ctranslate2.StorageView.from_array(dummy_features))
            logger.info("CUDA model and cublas verification successful.")
            return model, "cuda", "float16"
        except Exception as e:
            logger.warning(
                f"CUDA forward verification failed ({type(e).__name__}: {e}). "
                "GPU runtime libraries (e.g. cublas64_12.dll) not available. Falling back to multi-threaded CPU."
            )

    # 2. CPU fallback (int8, 8 threads)
    logger.info(f"Loading WhisperModel('{model_size}') on CPU (int8, multi-threaded)...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=8, download_root=str(MODELS_DIR))
    logger.info("CPU model loaded successfully.")
    return model, "cpu", "int8"

def transcribe_audio(
    audio_path: str,
    model_size: str = "tiny",
    language: Optional[str] = None,
) -> TranscriptResult:
    """Transcribes audio using Faster-Whisper with word-level timestamps."""
    path = Path(audio_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model, device_used, compute_type = _probe_and_load_whisper_model(model_size)

    logger.info(f"Starting transcription of '{path.name}' using {device_used} ({compute_type})...")
    start_time = time.time()

    try:
        segments_gen, info = model.transcribe(
            str(path),
            language=language,
            word_timestamps=True,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        # Force evaluation of segments generator
        raw_segments = list(segments_gen)
    except Exception as e:
        if device_used != "cpu":
            logger.warning(f"Inference failed on {device_used} ({e}). Retrying with CPU int8 fallback...")
            from faster_whisper import WhisperModel
            model = WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=8, download_root=str(MODELS_DIR))
            device_used = "cpu"
            compute_type = "int8"
            segments_gen, info = model.transcribe(
                str(path),
                language=language,
                word_timestamps=True,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )
            raw_segments = list(segments_gen)
        else:
            raise

    segment_objects: List[TranscriptSegment] = []
    seg_id = 0

    for seg in raw_segments:
        word_objs: List[WordTimestamp] = []
        if seg.words:
            for w in seg.words:
                word_objs.append(
                    WordTimestamp(
                        word=w.word.strip(),
                        start=round(w.start, 3),
                        end=round(w.end, 3),
                        probability=round(w.probability, 3),
                    )
                )

        segment_objects.append(
            TranscriptSegment(
                id=seg_id,
                start=round(seg.start, 3),
                end=round(seg.end, 3),
                text=seg.text.strip(),
                words=word_objs,
                avg_logprob=round(seg.avg_logprob, 3) if seg.avg_logprob is not None else None,
                no_speech_prob=round(seg.no_speech_prob, 3) if seg.no_speech_prob is not None else None,
            )
        )
        seg_id += 1

    elapsed = time.time() - start_time
    logger.info(
        f"Transcription complete in {elapsed:.2f}s. "
        f"Language detected: {info.language} (p={info.language_probability:.2f}), "
        f"Total segments: {len(segment_objects)}"
    )

    return TranscriptResult(
        language=info.language,
        duration_seconds=round(info.duration, 3),
        device_used=device_used,
        compute_type=compute_type,
        model_name=model_size,
        segments=segment_objects,
    )

def _format_srt_timestamp(seconds: float) -> str:
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{ms:03d}"

def export_transcript_srt(transcript: TranscriptResult, output_file: str) -> str:
    """Exports TranscriptResult to standard SRT subtitle file."""
    out_path = Path(output_file).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for i, seg in enumerate(transcript.segments, start=1):
        lines.append(str(i))
        lines.append(f"{_format_srt_timestamp(seg.start)} --> {_format_srt_timestamp(seg.end)}")
        lines.append(seg.text)
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return str(out_path)

def export_transcript_json(transcript: TranscriptResult, output_file: str) -> str:
    """Exports TranscriptResult to JSON file."""
    out_path = Path(output_file).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(transcript.model_dump(), indent=2), encoding="utf-8")
    return str(out_path)
