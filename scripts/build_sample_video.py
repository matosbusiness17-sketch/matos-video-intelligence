"""Builds an authorized local sample video with synchronized speech and rich visual slides."""
from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from matos_video.config import FFMPEG_PATH

def create_slide(title: str, subtitle: str, visual_elements: list, bg_color=(20, 24, 33)) -> np.ndarray:
    width, height = 1920, 1080
    img = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Accent bar at top
    draw.rectangle([0, 0, width, 12], fill=(41, 128, 255))

    # Header / Title
    draw.text((100, 100), title, fill=(255, 255, 255))
    draw.text((100, 180), subtitle, fill=(160, 175, 200))

    # Decorative frame / card
    draw.rounded_rectangle([100, 260, width - 100, height - 120], radius=24, outline=(50, 60, 80), width=3)

    # Render specific visual elements (tables, charts, badges, labels)
    for elem in visual_elements:
        elem_type = elem.get("type")
        if elem_type == "badge":
            x, y, text, color = elem["x"], elem["y"], elem["text"], elem["color"]
            draw.rounded_rectangle([x, y, x + 380, y + 60], radius=12, fill=color)
            draw.text((x + 20, y + 18), text, fill=(255, 255, 255))
        elif elem_type == "metric_box":
            x, y, label, val, sub = elem["x"], elem["y"], elem["label"], elem["value"], elem["sub"]
            draw.rounded_rectangle([x, y, x + 400, y + 260], radius=16, fill=(30, 36, 50), outline=(70, 85, 115), width=2)
            draw.text((x + 30, y + 30), label, fill=(160, 180, 210))
            draw.text((x + 30, y + 80), val, fill=(41, 200, 255))
            draw.text((x + 30, y + 180), sub, fill=(140, 150, 170))
        elif elem_type == "diagram_node":
            x, y, label, tech = elem["x"], elem["y"], elem["label"], elem["tech"]
            draw.rounded_rectangle([x, y, x + 320, y + 180], radius=16, fill=(28, 34, 48), outline=(60, 130, 240), width=3)
            draw.text((x + 30, y + 40), label, fill=(255, 255, 255))
            draw.text((x + 30, y + 100), tech, fill=(70, 210, 140))

    # Watermark
    draw.text((width - 450, height - 80), "MATOS VIDEO INTELLIGENCE LAB", fill=(100, 110, 130))

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def build_video():
    out_dir = Path("test_media")
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_path = out_dir / "sample_speech.wav"
    output_mp4 = out_dir / "matos_sample_talk.mp4"

    # Slide 1: Welcome & Overview (0 - 6.5s)
    slide1 = create_slide(
        title="MATOS VIDEO INTELLIGENCE",
        subtitle="Automated Local AI Analysis & Content Repurposing Toolkit",
        visual_elements=[
            {"type": "badge", "x": 160, "y": 320, "text": "STATUS: PRODUCTION READY", "color": (20, 140, 80)},
            {"type": "metric_box", "x": 160, "y": 420, "label": "TARGET ARCHITECTURE", "value": "LOCAL-FIRST", "sub": "Zero-Cloud Dependence"},
            {"type": "metric_box", "x": 620, "y": 420, "label": "COMPUTE STACK", "value": "RTX 5070 / i7", "sub": "Blackwell + 20-Core Ultra"},
            {"type": "metric_box", "x": 1080, "y": 420, "label": "PIPELINE GOAL", "value": "n8n AUTOMATION", "sub": "Autonomous Shorts Studio"},
        ]
    )

    # Slide 2: Visual Evidence - Chart and Metrics (6.5s - 14.0s)
    slide2 = create_slide(
        title="AUDIENCE ENGAGEMENT ANALYSIS",
        subtitle="Visual Evidence: Spoken words mention 45%, but chart proves retention factors",
        visual_elements=[
            {"type": "badge", "x": 160, "y": 320, "text": "METRIC ID: ENG-942", "color": (180, 80, 20)},
            {"type": "metric_box", "x": 160, "y": 420, "label": "CONVERSION LIFT", "value": "+45.2%", "sub": "Verified Spoken Evidence"},
            {"type": "metric_box", "x": 620, "y": 420, "label": "AVERAGE RETENTION", "value": "82.4%", "sub": "UNSPOKEN VISUAL DATA"},
            {"type": "metric_box", "x": 1080, "y": 420, "label": "DROP-OFF REDUCTION", "value": "-64.0%", "sub": "UNSPOKEN VISUAL DATA"},
        ]
    )

    # Slide 3: 3-Second Hook Retention Drop-Off (14.0s - 22.0s)
    slide3 = create_slide(
        title="RETENTION OPTIMIZATION & HOOK DETECTION",
        subtitle="Algorithmic Clip Selection: Coherent boundaries & hook detection",
        visual_elements=[
            {"type": "badge", "x": 160, "y": 320, "text": "CRITICAL ATTENTION ZONE", "color": (200, 30, 40)},
            {"type": "metric_box", "x": 160, "y": 420, "label": "HOOK WINDOW", "value": "0.0s - 3.0s", "sub": "Critical Viewer Gate"},
            {"type": "metric_box", "x": 620, "y": 420, "label": "OPTIMAL CLIP LENGTH", "value": "30 - 45 sec", "sub": "Shorts / Reels Target"},
            {"type": "metric_box", "x": 1080, "y": 420, "label": "VERTICAL RE-FRAME", "value": "9:16 (1080x1920)", "sub": "Subject-Preserving Crop"},
        ]
    )

    # Slide 4: End-to-End n8n Architecture (22.0s - 32.0s)
    slide4 = create_slide(
        title="AUTONOMOUS CONTENT FACTORY PIPELINE",
        subtitle="Self-Hosted n8n Workflow Integration Schema",
        visual_elements=[
            {"type": "badge", "x": 160, "y": 320, "text": "NODE WORKFLOW READY", "color": (30, 100, 200)},
            {"type": "diagram_node", "x": 160, "y": 460, "label": "1. AUDIO SPEECH", "tech": "Faster-Whisper"},
            {"type": "diagram_node", "x": 540, "y": 460, "label": "2. VISUAL FRAMES", "tech": "OpenCV Scene"},
            {"type": "diagram_node", "x": 920, "y": 460, "label": "3. MULTIMODAL AI", "tech": "Antigravity / Vision"},
            {"type": "diagram_node", "x": 1300, "y": 460, "label": "4. EXPORT BUNDLE", "tech": "JSON / MP4 / Report"},
        ]
    )

    # Render video stream using OpenCV VideoWriter
    temp_raw_avi = out_dir / "temp_video.avi"
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(str(temp_raw_avi), fourcc, fps, (1920, 1080))

    duration_total = 32.0  # seconds
    total_frames = int(duration_total * fps)

    for i in range(total_frames):
        t = i / fps
        if t < 6.5:
            frame = slide1
        elif t < 14.0:
            frame = slide2
        elif t < 22.0:
            frame = slide3
        else:
            frame = slide4

        # Add a subtle animated progress bar at bottom of card
        frame_copy = frame.copy()
        progress_width = int((t / duration_total) * (1920 - 200))
        cv2.rectangle(frame_copy, (100, 1080 - 140), (100 + progress_width, 1080 - 130), (41, 128, 255), -1)
        # Add timestamp HUD in corner
        cv2.putText(frame_copy, f"REC TIME: {t:05.2f}s | FRAME {i:04d}", (1450, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 140, 170), 2)
        out.write(frame_copy)

    out.release()
    print("Raw video track written to temporary AVI.")

    # Mux with audio using FFmpeg
    cmd = [
        FFMPEG_PATH,
        "-y",
        "-i", str(temp_raw_avi),
        "-i", str(audio_path),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(output_mp4),
    ]

    print(f"Muxing video + audio into '{output_mp4}'...")
    subprocess.run(cmd, check=True)
    temp_raw_avi.unlink(missing_ok=True)
    print(f"Successfully generated authorized sample video: {output_mp4} ({output_mp4.stat().st_size} bytes)")

if __name__ == "__main__":
    build_video()
