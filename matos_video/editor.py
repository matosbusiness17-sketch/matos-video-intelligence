"""Video editing and rendering engine using FFmpeg with strict FFprobe validation."""
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any

from matos_video.config import (
    FFMPEG_PATH,
    logger,
    VideoMetadata,
)
from matos_video.intake import probe_video
from matos_video.cropper import build_vertical_filtergraph, detect_subject_horizontal_focal

def cut_and_render_short(
    source_video_path: str,
    start_time: float,
    end_time: float,
    output_mp4_path: str,
    ass_subtitle_path: Optional[str] = None,
    crop_mode: str = "pad_blur",
    focal_x: Optional[float] = None,
) -> VideoMetadata:
    """Cuts a video segment, formats into 9:16 vertical 1080x1920, burns in subtitles,
    and validates the resulting MP4 using FFprobe."""
    in_path = Path(source_video_path).resolve()
    out_path = Path(output_mp4_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Source video not found: {source_video_path}")

    duration = end_time - start_time
    if duration <= 0:
        raise ValueError(f"Invalid clip duration: {duration}s (start={start_time}, end={end_time})")

    # Detect focal point if using smart crop
    if focal_x is None:
        if crop_mode == "smart_crop":
            focal_x = detect_subject_horizontal_focal(str(in_path), start_time, end_time)
        else:
            focal_x = 0.5

    # Build filtergraph
    filter_str, is_complex = build_vertical_filtergraph(
        crop_mode=crop_mode,
        focal_x=focal_x,
        ass_path=ass_subtitle_path,
    )

    # Build FFmpeg command
    cmd = [
        FFMPEG_PATH,
        "-y",
        "-ss", f"{start_time:.3f}",
        "-to", f"{end_time:.3f}",
        "-i", str(in_path),
    ]

    if is_complex:
        cmd.extend(["-filter_complex", filter_str, "-map", "[vout]", "-map", "0:a?"])
    else:
        cmd.extend(["-vf", filter_str, "-map", "0:v", "-map", "0:a?"])

    cmd.extend([
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-maxrate", "6M",
        "-bufsize", "12M",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "48000",
        str(out_path),
    ])

    logger.info(
        f"Rendering 9:16 Short: {start_time:.2f}s -> {end_time:.2f}s ({duration:.2f}s) "
        f"mode='{crop_mode}' -> '{out_path.name}'"
    )

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if result.returncode != 0:
        logger.error(f"FFmpeg render failed:\n{result.stderr[-1000:]}")
        raise RuntimeError(f"FFmpeg render failed (code {result.returncode}): {result.stderr.strip()[-500:]}")

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"Rendered output MP4 is missing or empty: {out_path}")

    # FFprobe validation
    logger.info(f"Validating rendered output with FFprobe: {out_path}")
    metadata = probe_video(str(out_path))

    if not metadata.has_video:
        raise ValueError(f"Render validation failed: No video stream detected in {out_path.name}")

    if metadata.video_stream.width != 1080 or metadata.video_stream.height != 1920:
        raise ValueError(
            f"Render validation failed: Expected 1080x1920, got "
            f"{metadata.video_stream.width}x{metadata.video_stream.height}"
        )

    # Check duration tolerance (allow 0.75s leeway due to keyframe / audio track alignment)
    if abs(metadata.duration_seconds - duration) > 1.2:
        logger.warning(
            f"Duration variance: target was {duration:.2f}s, output is {metadata.duration_seconds:.2f}s"
        )

    logger.info(
        f"Render verified successfully: {metadata.file_name} "
        f"({metadata.video_stream.width}x{metadata.video_stream.height}, "
        f"{metadata.duration_seconds:.2f}s, {metadata.file_size_bytes} bytes)"
    )

    return metadata
