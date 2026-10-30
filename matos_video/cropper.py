"""Intelligent 9:16 vertical re-framing module.

Supports non-destructive ambient-pad framing (recommended for 16:9 slides/podcasts),
zoom-blur framing, elevated card layout, and center cropping (for portrait talking heads).

NOTE: The smart_crop mode uses an OpenCV edge-gradient energy centroid heuristic.
Neural/ML facial landmark tracking is NOT implemented in this phase.
"""
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np

from matos_video.config import logger

def detect_subject_horizontal_focal(video_path: str, start_time: float, end_time: float, samples: int = 5) -> float:
    """Calculates horizontal center of visual activity using OpenCV edge-gradient energy centroid.
    NOTE: This is an edge-gradient heuristic, NOT neural AI face tracking.
    Defaults to 0.5 (center) if video cannot be opened or lacks prominent edges."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.5

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    sample_times = np.linspace(start_time, end_time, samples + 2)[1:-1]
    focal_points = []

    for t in sample_times:
        target_f = int(t * fps)
        if target_f >= total_frames:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_f)
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert to grayscale and detect edge energy
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        m = cv2.moments(edges)
        if m["m00"] > 0:
            cx = m["m10"] / m["m00"]
            focal_points.append(cx / float(frame.shape[1]))

    cap.release()

    if focal_points:
        avg_focal = float(np.mean(focal_points))
        # Clamp to avoid extreme edge cuts (keep within 0.25 to 0.75)
        clamped = max(0.25, min(0.75, avg_focal))
        logger.info(f"Detected edge-energy focal point: {clamped:.3f} (raw: {avg_focal:.3f})")
        return round(clamped, 3)

    return 0.5

def build_vertical_filtergraph(
    crop_mode: str = "pad_blur",
    focal_x: float = 0.5,
    ass_path: Optional[str] = None,
) -> Tuple[str, bool]:
    """Constructs the FFmpeg filter string for 9:16 vertical 1080x1920 video composition.

    Modes:
      - 'pad_blur': Centered 16:9 video with enhanced ambient blurred background.
                    Zero edge clipping, clean boundary stroke, rich vibrance.
      - 'pad_elevated': Elevated 16:9 video card (Y=420) leaving lower third for captions.
      - 'zoom_blur': 1.25x scaled foreground video (1080x760) over ambient blur.
                     Larger presence, mild 10% edge crop.
      - 'center': Full-bleed 1080x1920 crop. Discards 68.4% of widescreen width.
                  Recommended ONLY for centered single-speaker portrait/talking head footage.
      - 'smart_crop': Full-bleed 1080x1920 crop biased toward edge-energy centroid.

    Returns (filter_string, is_complex_filter)."""
    # Windows path escaping for ass filter: forward slashes, colon escaped
    ass_filter = ""
    if ass_path:
        escaped_ass = Path(ass_path).resolve().as_posix().replace(":", r"\:")
        ass_filter = f",ass='{escaped_ass}'"

    # Ambient background filter: scale to fill 1080x1920, blur, boost saturation and lift shadows
    # Eliminates the pitch-black void by avoiding artificial darkening
    ambient_bg = (
        "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,"
        "boxblur=30:5,eq=saturation=1.4:contrast=1.1:brightness=0.04[bg];"
    )

    if crop_mode == "pad_blur":
        # Full width 1080, centered vertically at (H-h)/2, subtle border to define video card
        filter_complex = (
            f"{ambient_bg}"
            "[0:v]scale=1080:-1,drawbox=x=0:y=0:w=iw:h=ih:color=white@0.25:t=2[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2{ass_filter}[vout]"
        )
        return filter_complex, True

    elif crop_mode == "pad_elevated":
        # Full width 1080, positioned in upper half (Y=420), leaving dedicated reading zone below
        filter_complex = (
            f"{ambient_bg}"
            "[0:v]scale=1080:-1,drawbox=x=0:y=0:w=iw:h=ih:color=white@0.3:t=2[fg];"
            f"[bg][fg]overlay=(W-w)/2:420{ass_filter}[vout]"
        )
        return filter_complex, True

    elif crop_mode == "zoom_blur":
        # Scale foreground to 1.25x width (1350px), crop to 1080, height is ~760px (40% of screen)
        # Moderate zoom that increases subject size while preserving 80% of horizontal content
        filter_complex = (
            f"{ambient_bg}"
            "[0:v]scale=1350:-1,crop=1080:ih:(in_w-1080)/2:0,"
            "drawbox=x=0:y=0:w=iw:h=ih:color=white@0.25:t=2[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2{ass_filter}[vout]"
        )
        return filter_complex, True

    elif crop_mode in ["center", "crop", "smart_crop"]:
        # Direct vertical crop:
        # Scale video height to 1920 (width becomes 3413px for 16:9), crop 1080 width
        use_focal = focal_x if crop_mode in ["crop", "smart_crop"] else 0.5
        simple_vf = (
            f"scale=-1:1920,"
            f"crop=1080:1920:(in_w-1080)*{use_focal:.3f}:0"
            f"{ass_filter}"
        )
        return simple_vf, False

    else:
        raise ValueError(
            f"Unknown crop_mode: '{crop_mode}'. Choose 'pad_blur', 'pad_elevated', 'zoom_blur', 'center', or 'smart_crop'."
        )
