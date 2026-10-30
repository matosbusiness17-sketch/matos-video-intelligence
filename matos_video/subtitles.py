"""Subtitle generation module producing styled ASS subtitles for 9:16 vertical Shorts."""
from pathlib import Path
from typing import List, Optional

from matos_video.config import (
    TranscriptResult,
    TranscriptSegment,
    WordTimestamp,
    logger,
)

def format_ass_timestamp(seconds: float) -> str:
    """Formats seconds into ASS timestamp format: H:MM:SS.cs"""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    if cs >= 100:
        secs += 1
        cs = 0
    return f"{hrs:d}:{mins:02d}:{secs:02d}.{cs:02d}"

def generate_ass_subtitles(
    transcript: TranscriptResult,
    start_time: float,
    end_time: float,
    output_ass_path: str,
    font_name: str = "Arial",
    font_size: int = 56,
    primary_color: str = "&H00FFFFFF",      # Crisp White
    highlight_color: str = "&H0000FFFF",    # Vibrant Yellow
    outline_color: str = "&H00000000",      # Deep Black
    margin_v: int = 280,                    # Safe zone: above bottom 240px
) -> str:
    """Generates an Advanced SubStation Alpha (.ass) subtitle file with word-level highlight
    tailored for 1080x1920 vertical Shorts."""
    out_path = Path(output_ass_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},{primary_color},&H000000FF,{outline_color},&H80000000,-1,0,0,0,100,100,0,0,1,5,2,2,60,60,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    events: List[str] = []

    # Filter segments within clip window
    clip_segments = [
        s for s in transcript.segments
        if s.end > start_time and s.start < end_time
    ]

    for seg in clip_segments:
        # Relative timestamps offset by clip start_time
        seg_start_rel = max(0.0, seg.start - start_time)
        seg_end_rel = min(end_time - start_time, seg.end - start_time)

        if seg_end_rel <= seg_start_rel:
            continue

        if seg.words and len(seg.words) > 0:
            # Word-level animation: render word-by-word active highlighting
            words = [w for w in seg.words if w.end > start_time and w.start < end_time]
            if not words:
                continue

            # Group words into short chunks of 3-5 words to avoid multi-line overflow
            chunk_size = 4
            for i in range(0, len(words), chunk_size):
                chunk = words[i : i + chunk_size]
                chunk_start = max(0.0, chunk[0].start - start_time)
                chunk_end = min(end_time - start_time, chunk[-1].end - start_time)

                if chunk_end <= chunk_start:
                    continue

                # Each word in the chunk highlights in sequence
                for active_idx, w in enumerate(chunk):
                    w_start_rel = max(chunk_start, w.start - start_time)
                    w_end_rel = min(chunk_end, w.end - start_time)

                    # Build styled line: active word has highlight_color
                    line_parts = []
                    for idx, cw in enumerate(chunk):
                        if idx == active_idx:
                            line_parts.append(f"{{\\c{highlight_color}\\b1}}{cw.word}{{\\c{primary_color}\\b0}}")
                        else:
                            line_parts.append(cw.word)

                    dialogue_text = " ".join(line_parts)
                    start_str = format_ass_timestamp(w_start_rel)
                    end_str = format_ass_timestamp(w_end_rel)
                    events.append(f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{dialogue_text}")
        else:
            # Fallback to segment-level caption
            start_str = format_ass_timestamp(seg_start_rel)
            end_str = format_ass_timestamp(seg_end_rel)
            text_clean = seg.text.strip().replace("\n", " ")
            events.append(f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{text_clean}")

    content = header + "\n".join(events) + "\n"
    out_path.write_text(content, encoding="utf-8")
    logger.info(f"Generated ASS subtitles with {len(events)} cues -> {out_path}")
    return str(out_path)
