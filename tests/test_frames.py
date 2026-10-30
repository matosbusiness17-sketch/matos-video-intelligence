"""Tests for frame extraction and scene detection."""
import pytest
from pathlib import Path
from matos_video.frames import (
    extract_frames,
    extract_frame_at_timestamp,
    compute_frame_difference,
)
import numpy as np

SAMPLE_VIDEO = Path("test_media/matos_sample_talk.mp4")

def test_compute_frame_difference():
    # Identical frames -> 0.0
    f1 = np.zeros((100, 100, 3), dtype=np.uint8)
    f2 = np.zeros((100, 100, 3), dtype=np.uint8)
    assert compute_frame_difference(f1, f2) == 0.0

    # Completely different frames -> high score
    f3 = np.ones((100, 100, 3), dtype=np.uint8) * 255
    score = compute_frame_difference(f1, f3)
    assert score > 0.5

def test_extract_frames(tmp_path):
    frames_dir = tmp_path / "frames"
    frames = extract_frames(
        str(SAMPLE_VIDEO),
        str(frames_dir),
        interval_seconds=4.0,
        scene_threshold=0.03,
    )
    assert len(frames) >= 7
    manifest = frames_dir / "frames_manifest.json"
    assert manifest.exists()
    for f in frames:
        assert Path(f.file_path).exists()
        assert f.width > 0
        assert f.height > 0

def test_targeted_frame_extraction(tmp_path):
    target_img = tmp_path / "targeted.jpg"
    out = extract_frame_at_timestamp(str(SAMPLE_VIDEO), 10.0, str(target_img))
    assert Path(out).exists()
    assert Path(out).stat().st_size > 10000
