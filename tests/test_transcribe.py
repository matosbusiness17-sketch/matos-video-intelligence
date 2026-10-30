"""Tests for audio extraction and Faster-Whisper transcription."""
import pytest
from pathlib import Path
from matos_video.transcribe import (
    extract_audio,
    transcribe_audio,
    export_transcript_json,
    export_transcript_srt,
)
from matos_video.config import TranscriptResult

SAMPLE_VIDEO = Path("test_media/matos_sample_talk.mp4")

def test_extract_audio(tmp_path):
    out_wav = tmp_path / "test_extract.wav"
    extracted = extract_audio(str(SAMPLE_VIDEO), str(out_wav))
    assert Path(extracted).exists()
    assert Path(extracted).stat().st_size > 500000

def test_transcribe_audio(tmp_path):
    out_wav = tmp_path / "audio_for_transcribe.wav"
    extract_audio(str(SAMPLE_VIDEO), str(out_wav))
    
    result = transcribe_audio(str(out_wav), model_size="tiny", language="en")
    assert isinstance(result, TranscriptResult)
    assert result.language == "en"
    assert len(result.segments) >= 4
    assert result.device_used in ["cuda", "cpu"]
    
    # Check word-level timestamps
    first_seg = result.segments[0]
    assert len(first_seg.words) > 0
    assert first_seg.words[0].end >= first_seg.words[0].start

    # Check exports
    json_path = export_transcript_json(result, str(tmp_path / "t.json"))
    srt_path = export_transcript_srt(result, str(tmp_path / "t.srt"))
    assert Path(json_path).exists()
    assert Path(srt_path).exists()
    assert "-->" in Path(srt_path).read_text(encoding="utf-8")
