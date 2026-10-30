"""Intelligent frame extraction and scene detection module using OpenCV and FFmpeg."""
import json
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np

from matos_video.config import (
    FFMPEG_PATH,
    logger,
    FrameInfo,
)

def compute_frame_difference(prev_frame: np.ndarray, curr_frame: np.ndarray) -> float:
    """Computes a robust scene difference score based on mean pixel delta.
    Static frames with minor motion produce scores < 0.01.
    Slide/graphic changes produce scores ~0.05 - 0.20.
    Camera/scene cuts produce scores > 0.30."""
    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if len(prev_frame.shape) == 3 else prev_frame
    gray_curr = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY) if len(curr_frame.shape) == 3 else curr_frame
    mean_abs_diff = float(np.mean(cv2.absdiff(gray_prev, gray_curr)))
    # Scale: 10.0 mean absdiff = 1.0 score
    score = min(1.0, mean_abs_diff / 10.0)
    return round(float(score), 4)

def extract_frames(
    video_path: str,
    output_dir: str,
    interval_seconds: float = 2.0,
    scene_threshold: float = 0.03,
    max_dimension: Optional[int] = 1280,
) -> List[FrameInfo]:
    """Extracts timestamped frames from video based on interval sampling and scene changes."""
    in_path = Path(video_path).resolve()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV failed to open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if total_frames > 0 else 0.0

    frame_interval = max(1, int(fps * interval_seconds))
    logger.info(
        f"Extracting frames from '{in_path.name}' (FPS={fps:.2f}, "
        f"total_frames={total_frames}, duration={duration:.2f}s, interval={interval_seconds}s)"
    )

    extracted_frames: List[FrameInfo] = []
    last_consecutive_small: Optional[np.ndarray] = None
    last_saved_time = -999.0
    min_scene_gap = 1.0  # Minimum 1.0s cooldown between detected scene change triggers
    frame_idx = 0
    saved_count = 0

    # For high-fps video, check scene change cadence every few frames to maximize speed
    check_cadence = max(1, int(fps * 0.25))  # check 4 times per second

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_sec = frame_idx / fps
        is_sample_point = (frame_idx % frame_interval == 0) or (frame_idx == 0)

        diff_score = 0.0
        is_scene_change = False

        if frame_idx % check_cadence == 0:
            h, w = frame.shape[:2]
            small_frame = cv2.resize(frame, (160, int(160 * h / w))) if w > 160 else frame
            if last_consecutive_small is not None:
                diff_score = compute_frame_difference(last_consecutive_small, small_frame)
                if diff_score >= scene_threshold and (timestamp_sec - last_saved_time) >= min_scene_gap:
                    is_scene_change = True
            last_consecutive_small = small_frame

        if is_sample_point or is_scene_change:
            last_saved_time = timestamp_sec
            h, w = frame.shape[:2]
            # Resize if max_dimension is specified
            save_frame = frame
            if max_dimension and max(h, w) > max_dimension:
                scale = max_dimension / float(max(h, w))
                new_w, new_h = int(w * scale), int(h * scale)
                save_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            ts_ms = int(timestamp_sec * 1000)
            tag = "_scene" if is_scene_change else ""
            filename = f"frame_{saved_count:04d}_{ts_ms:07d}ms{tag}.jpg"
            file_path = out_dir / filename
            cv2.imwrite(str(file_path), save_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])

            info = FrameInfo(
                frame_index=frame_idx,
                timestamp_seconds=round(timestamp_sec, 3),
                file_path=str(file_path),
                relative_path=filename,
                width=save_frame.shape[1],
                height=save_frame.shape[0],
                is_scene_change=is_scene_change,
                scene_score=diff_score,
            )
            extracted_frames.append(info)
            saved_count += 1

        frame_idx += 1

    cap.release()
    logger.info(f"Extracted {len(extracted_frames)} frames into '{out_dir.name}'")

    # Save manifest
    manifest_file = out_dir / "frames_manifest.json"
    manifest_data = [f.model_dump() for f in extracted_frames]
    manifest_file.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

    return extracted_frames

def extract_frame_at_timestamp(
    video_path: str,
    timestamp_sec: float,
    output_path: str,
) -> str:
    """Extracts a precise frame at a given timestamp using FFmpeg."""
    in_path = Path(video_path).resolve()
    out_path = Path(output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        FFMPEG_PATH,
        "-y",
        "-ss", f"{timestamp_sec:.3f}",
        "-i", str(in_path),
        "-vframes", "1",
        "-q:v", "2",
        str(out_path),
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg frame extraction failed at {timestamp_sec}s: {result.stderr.strip()}")

    if not out_path.exists():
        raise RuntimeError(f"Target frame was not created: {out_path}")

    return str(out_path)
