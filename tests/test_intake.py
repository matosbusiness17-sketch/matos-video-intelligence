"""Tests for video intake and metadata probing."""
import pytest
from pathlib import Path
from matos_video.intake import probe_video
from matos_video.config import VideoMetadata

SAMPLE_VIDEO = Path("test_media/matos_sample_talk.mp4")

def test_probe_video_success():
    assert SAMPLE_VIDEO.exists(), "Sample video must exist for tests"
    meta = probe_video(str(SAMPLE_VIDEO))
    assert isinstance(meta, VideoMetadata)
    assert meta.has_video is True
    assert meta.has_audio is True
    assert meta.duration_seconds > 30.0
    assert meta.video_stream.width == 1920
    assert meta.video_stream.height == 1080
    assert meta.video_stream.fps == 30.0
    assert meta.audio_stream.sample_rate > 0

def test_probe_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        probe_video("non_existent_file_xyz.mp4")
