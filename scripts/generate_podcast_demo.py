"""Builds a 72-second authorized local video with two distinct 25-45s topics for Shorts rendering tests."""
from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageDraw

from matos_video.config import FFMPEG_PATH

def create_slide(title: str, subtitle: str, badge_text: str, badge_color: tuple, cards: list, bg_color=(20, 24, 33)) -> np.ndarray:
    width, height = 1920, 1080
    img = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Accent line
    draw.rectangle([0, 0, width, 12], fill=(41, 128, 255))

    # Header
    draw.text((100, 90), title, fill=(255, 255, 255))
    draw.text((100, 170), subtitle, fill=(160, 175, 200))

    # Badge
    draw.rounded_rectangle([100, 240, 520, 300], radius=12, fill=badge_color)
    draw.text((120, 258), badge_text, fill=(255, 255, 255))

    # Cards layout
    card_w = 400
    card_h = 280
    start_x = 100
    start_y = 340
    gap = 40

    for idx, card in enumerate(cards):
        x = start_x + idx * (card_w + gap)
        if x + card_w > width:
            break
        draw.rounded_rectangle([x, start_y, x + card_w, start_y + card_h], radius=16, fill=(30, 36, 50), outline=(70, 85, 115), width=2)
        draw.text((x + 25, start_y + 30), card.get("label", ""), fill=(160, 180, 210))
        draw.text((x + 25, start_y + 80), card.get("val", ""), fill=(41, 200, 255))
        draw.text((x + 25, start_y + 190), card.get("sub", ""), fill=(140, 150, 170))

    # Footer
    draw.text((width - 450, height - 70), "MATOS VIDEO INTELLIGENCE LAB", fill=(100, 110, 130))

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def generate_podcast_video():
    out_dir = Path("test_media")
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_path = out_dir / "podcast_speech.wav"
    output_mp4 = out_dir / "matos_podcast_demo.mp4"

    # 1. Generate 70s TTS speech via PowerShell SAPI
    ps_script = out_dir / "gen_speech.ps1"
    ps_content = """Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SetOutputToWaveFile("test_media/podcast_speech.wav")

$script = @"
Welcome to episode forty two of the Matos Systems Engineering Podcast.
In our first segment, we analyze how viewer retention changes in short form video.
Notice that the chart on your screen shows an eighty two percent average retention rate.
The key to retaining short form viewers is an irresistible hook in the first three seconds.
Without an immediate visual or auditory hook, over sixty percent of viewers swipe away before second five.
Now let us transition to our second segment on autonomous infrastructure.
Our content factory relies on self-hosted n8n workflows paired with local machine learning models.
We extract word-level transcripts with Faster-Whisper and analyze keyframes with OpenCV.
Because our entire pipeline runs locally on Windows, processing costs remain zero dollars with complete privacy.
Thank you for watching this technical demonstration of Matos Video Intelligence.
"@

$synth.Speak($script)
$synth.Dispose()
"""
    ps_script.write_text(ps_content, encoding="utf-8")
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps_script)], check=True)
    ps_script.unlink(missing_ok=True)
    print("Podcast speech audio generated.")

    # 2. Get duration of speech
    cap_audio = cv2.VideoCapture(str(audio_path))
    # Or use ffprobe to get duration
    from matos_video.intake import probe_video
    meta = probe_video(str(audio_path))
    duration_total = meta.duration_seconds
    print(f"Speech duration: {duration_total:.2f}s")

    # 3. Slides definitions
    # Slide 1: Podcast Intro (0.0s - 14.0s)
    slide1 = create_slide(
        title="MATOS SYSTEMS ENGINEERING PODCAST",
        subtitle="Episode 42: Retention Optimization & Automated Shorts Editing",
        badge_text="EPISODE 42 | AI AWARENESS",
        badge_color=(20, 120, 220),
        cards=[
            {"label": "SERIES", "val": "AI AUTOMATION", "sub": "Engineering Cycle 2026"},
            {"label": "HOST", "val": "MATOS AI LAB", "sub": "Local-First Architecture"},
            {"label": "TARGET", "val": "DUAL SHORTS", "sub": "25-45s Optimized Clips"},
        ]
    )

    # Slide 2: Retention Analytics (14.0s - 34.0s) - End of Topic A
    slide2 = create_slide(
        title="SHORT-FORM RETENTION ANALYTICS",
        subtitle="Audience drop-off analysis: Hook importance in the first 3 seconds",
        badge_text="METRIC ID: RET-882",
        badge_color=(220, 40, 50),
        cards=[
            {"label": "AVERAGE RETENTION", "val": "82.4%", "sub": "UNSPOKEN: Top 10% Benchmark"},
            {"label": "SWIPE-AWAY RATE", "val": "-64.0%", "sub": "UNSPOKEN: Reduced Drop-Off"},
            {"label": "HOOK WINDOW", "val": "0.0s - 3.0s", "sub": "Critical Viewer Gate"},
        ]
    )

    # Slide 3: Autonomous n8n Workflow (34.0s - 54.0s) - Topic B start
    slide3 = create_slide(
        title="AUTONOMOUS n8n CONTENT FACTORY",
        subtitle="Self-Hosted Workflow Architecture for Unattended Processing",
        badge_text="PIPELINE: ACTIVE (v2.4)",
        badge_color=(30, 160, 90),
        cards=[
            {"label": "ORCHESTRATOR", "val": "n8n SELF-HOSTED", "sub": "UNSPOKEN: Docker Container"},
            {"label": "SPEECH ENGINE", "val": "FASTER-WHISPER", "sub": "CTranslate2 Local C++"},
            {"label": "FRAME ENGINE", "val": "OPENCV HEADLESS", "sub": "Automated Scene Detection"},
        ]
    )

    # Slide 4: Cost & Privacy Architecture (54.0s - End) - Topic B conclusion
    slide4 = create_slide(
        title="LOCAL INFRASTRUCTURE & PRIVACY",
        subtitle="Zero Cloud Dependence: Full Privacy & Zero Operating Costs",
        badge_text="SECURITY: 100% AIR-GAPPED READY",
        badge_color=(140, 40, 200),
        cards=[
            {"label": "OPERATING COST", "val": "$0.00 / MO", "sub": "UNSPOKEN: Zero Cloud Subscriptions"},
            {"label": "DATA PRIVACY", "val": "STRICT LOCAL", "sub": "Client Media Protected"},
            {"label": "RENDER SPEC", "val": "1080x1920 MP4", "sub": "9:16 Vertical Delivery"},
        ]
    )

    # 4. Render Video frames
    temp_avi = out_dir / "temp_podcast.avi"
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(str(temp_avi), fourcc, fps, (1920, 1080))
    total_frames = int(duration_total * fps)

    for i in range(total_frames):
        t = i / fps
        if t < 14.0:
            frame = slide1
        elif t < 34.0:
            frame = slide2
        elif t < 54.0:
            frame = slide3
        else:
            frame = slide4

        frame_copy = frame.copy()
        # Animated progress bar
        p_w = int((t / duration_total) * (1920 - 200))
        cv2.rectangle(frame_copy, (100, 1080 - 140), (100 + p_w, 1080 - 130), (41, 128, 255), -1)
        # HUD
        cv2.putText(frame_copy, f"REC: {t:05.2f}s | F: {i:04d}", (1500, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 140, 170), 2)
        out.write(frame_copy)

    out.release()
    print("Raw video track rendered.")

    # 5. Mux with audio
    cmd = [
        FFMPEG_PATH,
        "-y",
        "-i", str(temp_avi),
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
    subprocess.run(cmd, check=True)
    temp_avi.unlink(missing_ok=True)
    print(f"Generated authorized podcast sample: {output_mp4} ({output_mp4.stat().st_size} bytes)")

if __name__ == "__main__":
    generate_podcast_video()
