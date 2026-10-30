"""Automated tests for subtitle generation, cropping, and FFmpeg video cutting."""
import pytest
from pathlib import Path

from matos_video.config import TranscriptResult, TranscriptSegment, WordTimestamp
from matos_video.subtitles import generate_ass_subtitles, format_ass_timestamp
from matos_video.cropper import build_vertical_filtergraph, detect_subject_horizontal_focal
from matos_video.editor import cut_and_render_short
from matos_video.intake import probe_video

SAMPLE_VIDEO = Path("test_media/matos_sample_talk.mp4")

def test_format_ass_timestamp():
    assert format_ass_timestamp(0.0) == "0:00:00.00"
    assert format_ass_timestamp(65.45) == "0:01:05.45"
    assert format_ass_timestamp(3605.12) == "1:00:05.12"

def test_generate_ass_subtitles(tmp_path):
    transcript = TranscriptResult(
        language="en",
        duration_seconds=10.0,
        device_used="cpu",
        compute_type="int8",
        model_name="tiny",
        segments=[
            TranscriptSegment(
                id=0,
                start=1.0,
                end=5.0,
                text="Welcome to the video.",
                words=[
                    WordTimestamp(word="Welcome", start=1.0, end=2.0, probability=0.9),
                    WordTimestamp(word="to", start=2.0, end=2.5, probability=0.9),
                    WordTimestamp(word="the", start=2.5, end=3.0, probability=0.9),
                    WordTimestamp(word="video.", start=3.0, end=4.5, probability=0.9),
                ],
            )
        ],
    )
    ass_path = tmp_path / "subs.ass"
    out = generate_ass_subtitles(transcript, start_time=1.0, end_time=5.0, output_ass_path=str(ass_path))
    assert Path(out).exists()
    content = Path(out).read_text(encoding="utf-8")
    assert "PlayResX: 1080" in content
    assert "PlayResY: 1920" in content
    assert "Dialogue:" in content
    assert "&H0000FFFF" in content  # Highlight color present

def test_cropper_filtergraphs():
    # Pad blur complex filter
    pad_str, is_complex = build_vertical_filtergraph(crop_mode="pad_blur")
    assert is_complex is True
    assert "boxblur=30:5" in pad_str
    assert "scale=1080:1920" in pad_str
    assert "drawbox=" in pad_str

    # Pad elevated complex filter
    elev_str, is_elev_complex = build_vertical_filtergraph(crop_mode="pad_elevated")
    assert is_elev_complex is True
    assert "overlay=(W-w)/2:420" in elev_str

    # Zoom blur complex filter
    zoom_str, is_zoom_complex = build_vertical_filtergraph(crop_mode="zoom_blur")
    assert is_zoom_complex is True
    assert "scale=1350:-1" in zoom_str

    # Center crop simple filter
    center_str, is_comp2 = build_vertical_filtergraph(crop_mode="center")
    assert is_comp2 is False
    assert "crop=1080:1920" in center_str

    # Smart crop (edge-energy biased)
    smart_str, _ = build_vertical_filtergraph(crop_mode="smart_crop", focal_x=0.6)
    assert "crop=1080:1920" in smart_str
    assert "0.600" in smart_str

    with pytest.raises(ValueError):
        build_vertical_filtergraph(crop_mode="invalid_mode")

def test_cut_and_render_short_pad_blur(tmp_path):
    """Renders a 3-second test short in pad_blur mode and verifies resolution with FFprobe."""
    assert SAMPLE_VIDEO.exists()
    out_mp4 = tmp_path / "test_short_pad.mp4"
    meta = cut_and_render_short(
        source_video_path=str(SAMPLE_VIDEO),
        start_time=0.0,
        end_time=3.0,
        output_mp4_path=str(out_mp4),
        crop_mode="pad_blur",
    )
    assert Path(out_mp4).exists()
    assert meta.video_stream.width == 1080
    assert meta.video_stream.height == 1920
    assert meta.has_audio is True
    assert 2.5 <= meta.duration_seconds <= 3.5

def test_cut_and_render_short_center_crop(tmp_path):
    """Renders a 3-second test short in center crop mode and verifies resolution with FFprobe."""
    assert SAMPLE_VIDEO.exists()
    out_mp4 = tmp_path / "test_short_center.mp4"
    meta = cut_and_render_short(
        source_video_path=str(SAMPLE_VIDEO),
        start_time=5.0,
        end_time=8.0,
        output_mp4_path=str(out_mp4),
        crop_mode="center",
    )
    assert Path(out_mp4).exists()
    assert meta.video_stream.width == 1080
    assert meta.video_stream.height == 1920
    assert meta.has_audio is True
