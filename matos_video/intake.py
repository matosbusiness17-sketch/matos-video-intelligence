"""Video intake and metadata probing module using FFprobe and yt-dlp."""
import json
import os
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any

from matos_video.config import (
    FFPROBE_PATH,
    FFMPEG_PATH,
    logger,
    VideoMetadata,
    VideoStreamInfo,
    AudioStreamInfo,
)

def _parse_fps(fps_str: Optional[str]) -> float:
    if not fps_str or fps_str == "0/0":
        return 0.0
    try:
        if "/" in fps_str:
            num, den = fps_str.split("/", 1)
            den_f = float(den)
            return float(num) / den_f if den_f != 0 else 0.0
        return float(fps_str)
    except Exception:
        return 0.0

def probe_video(file_path: str) -> VideoMetadata:
    """Probes a local video file using ffprobe and returns structured VideoMetadata."""
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {file_path}")

    cmd = [
        FFPROBE_PATH,
        "-v", "error",
        "-show_format",
        "-show_streams",
        "-of", "json",
        str(path),
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"FFprobe failed (code {result.returncode}): {result.stderr.strip()}")

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse FFprobe JSON output: {e}\nRaw output: {result.stdout}")

    format_info = data.get("format", {})
    streams = data.get("streams", [])

    duration = float(format_info.get("duration", 0.0))
    file_size = int(format_info.get("size", path.stat().st_size))
    format_name = format_info.get("format_name", "unknown")

    video_stream_obj: Optional[VideoStreamInfo] = None
    audio_stream_obj: Optional[AudioStreamInfo] = None

    for s in streams:
        codec_type = s.get("codec_type")
        if codec_type == "video" and video_stream_obj is None:
            v_duration = float(s.get("duration", duration))
            fps = _parse_fps(s.get("r_frame_rate")) or _parse_fps(s.get("avg_frame_rate"))
            bitrate = int(s.get("bit_rate")) if s.get("bit_rate") else None
            video_stream_obj = VideoStreamInfo(
                codec_name=s.get("codec_name", "unknown"),
                width=int(s.get("width", 0)),
                height=int(s.get("height", 0)),
                fps=round(fps, 3),
                duration_seconds=round(v_duration, 3),
                bitrate=bitrate,
            )
        elif codec_type == "audio" and audio_stream_obj is None:
            a_duration = float(s.get("duration", duration))
            sample_rate = int(s.get("sample_rate", 0))
            channels = int(s.get("channels", 0))
            bitrate = int(s.get("bit_rate")) if s.get("bit_rate") else None
            audio_stream_obj = AudioStreamInfo(
                codec_name=s.get("codec_name", "unknown"),
                channels=channels,
                sample_rate=sample_rate,
                duration_seconds=round(a_duration, 3),
                bitrate=bitrate,
            )

    metadata = VideoMetadata(
        file_path=str(path),
        file_name=path.name,
        file_size_bytes=file_size,
        format_name=format_name,
        duration_seconds=round(duration, 3),
        video_stream=video_stream_obj,
        audio_stream=audio_stream_obj,
        has_video=video_stream_obj is not None,
        has_audio=audio_stream_obj is not None,
    )

    logger.info(
        f"Probed '{path.name}': duration={metadata.duration_seconds}s, "
        f"video={metadata.has_video} ({video_stream_obj.width}x{video_stream_obj.height} @ {video_stream_obj.fps}fps) "
        if video_stream_obj else "no-video, "
        f"audio={metadata.has_audio} ({audio_stream_obj.sample_rate}Hz {audio_stream_obj.channels}ch)"
        if audio_stream_obj else "no-audio"
    )

    return metadata

def download_video(url: str, output_dir: str) -> str:
    """Downloads an authorized video URL using yt-dlp to output_dir."""
    import yt_dlp

    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    ffmpeg_dir = str(Path(FFMPEG_PATH).parent)
    ydl_opts = {
        "format": "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": str(out_path / "%(title).50s [%(id)s].%(ext)s"),
        "restrictfilenames": True,
        "noplaylist": True,
        "quiet": False,
        "ffmpeg_location": ffmpeg_dir,
    }

    logger.info(f"Downloading authorized URL with yt-dlp: {url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        # Handle cases where container is merged to mp4 or mkv
        downloaded_file = Path(filename)
        if not downloaded_file.exists():
            # Check for alternative extensions
            for ext in [".mp4", ".mkv", ".webm"]:
                alt = downloaded_file.with_suffix(ext)
                if alt.exists():
                    downloaded_file = alt
                    break
        
        logger.info(f"Downloaded video successfully: {downloaded_file}")
        return str(downloaded_file)
