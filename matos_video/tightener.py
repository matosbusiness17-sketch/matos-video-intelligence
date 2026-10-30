"""Smart Tightening module for Matos Video Intelligence.

Implements silence trimming, filler/repetition removal, Edit Decision List (EDL) generation,
timeline-remapped subtitles, and multi-segment frame-accurate FFmpeg assembly.
"""
import datetime
import json
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from matos_video.config import (
    FFMPEG_PATH,
    logger,
    TranscriptResult,
    TranscriptSegment,
    WordTimestamp,
    EditDecision,
    TightenedSegment,
    EditDecisionList,
    VideoMetadata,
)
from matos_video.intake import probe_video
from matos_video.cropper import build_vertical_filtergraph, detect_subject_horizontal_focal
from matos_video.subtitles import format_ass_timestamp

def detect_silence_cuts(
    transcript: TranscriptResult,
    start_time: float,
    end_time: float,
    min_pause_duration: float = 0.50,
    retain_pause: float = 0.25,
) -> List[EditDecision]:
    """Detects pauses between words exceeding min_pause_duration and proposes shortening
    them to retain_pause, preserving natural speech cadences while eliminating dead air.
    """
    cuts: List[EditDecision] = []
    
    # Collect all words in the candidate window
    words_in_window: List[WordTimestamp] = []
    for seg in transcript.segments:
        for w in seg.words:
            if w.end > start_time and w.start < end_time:
                words_in_window.append(w)
    
    words_in_window.sort(key=lambda w: w.start)
    if len(words_in_window) < 2:
        return cuts
    
    cut_idx = 1
    for i in range(len(words_in_window) - 1):
        w_curr = words_in_window[i]
        w_next = words_in_window[i + 1]
        
        gap = w_next.start - w_curr.end
        if gap > min_pause_duration:
            # Cut excess pause from the center of the gap, keeping retain_pause/2 on each side
            pad = retain_pause / 2.0
            cut_start = round(w_curr.end + pad, 3)
            cut_end = round(w_next.start - pad, 3)
            dur_cut = round(cut_end - cut_start, 3)
            
            if dur_cut > 0.05 and cut_start >= start_time and cut_end <= end_time:
                cuts.append(EditDecision(
                    cut_id=f"silence_{cut_idx:02d}",
                    cut_type="silence_trim",
                    original_start=cut_start,
                    original_end=cut_end,
                    duration_cut=dur_cut,
                    spoken_content=None,
                    is_meaningful_content=False,
                    rationale=(
                        f"Shorten {gap:.2f}s gap between '{w_curr.word.strip()}' and '{w_next.word.strip()}' "
                        f"to natural {retain_pause:.2f}s pause (saving {dur_cut:.2f}s)."
                    ),
                    approved=True,  # Mechanical silence trims do not alter spoken words
                ))
                cut_idx += 1
                
    return cuts

def detect_filler_and_repetition(
    transcript: TranscriptResult,
    start_time: float,
    end_time: float,
) -> List[EditDecision]:
    """Identifies unnecessary filler words, stumbles, and repetitive phrases within the clip window.
    These cuts represent meaningful spoken content removal and STRICTLY require human operator approval.
    """
    cuts: List[EditDecision] = []
    
    # Collect segments within the candidate window
    active_segs = [s for s in transcript.segments if s.end > start_time and s.start < end_time]
    
    # Common speech filler repetitions to scan for
    cut_counter = 1
    for seg in active_segs:
        words = [w for w in seg.words if w.end > start_time and w.start < end_time]
        if len(words) < 3:
            continue
            
        # Pattern 1: Scan for adjacent phrase repetition (e.g. "that place of... and that place of like")
        # Check 3-word or 4-word n-grams
        for n in range(3, 5):
            for i in range(len(words) - (2 * n)):
                phrase1 = " ".join(w.word.strip().lower() for w in words[i : i + n])
                phrase2 = " ".join(w.word.strip().lower() for w in words[i + n : i + (2 * n)])
                
                # Check for significant overlap or stumble repetition
                if phrase1 == phrase2 or (phrase1 in phrase2) or ("that place of" in phrase1 and "that place of" in phrase2):
                    c_start = round(words[i + n].start, 3)
                    c_end = round(words[i + (2 * n)].end, 3)
                    spoken = " ".join(w.word.strip() for w in words[i + n : i + (2 * n)])
                    
                    cuts.append(EditDecision(
                        cut_id=f"repetition_{cut_counter:02d}",
                        cut_type="repetition_removal",
                        original_start=c_start,
                        original_end=c_end,
                        duration_cut=round(c_end - c_start, 3),
                        spoken_content=spoken,
                        is_meaningful_content=True,
                        rationale=f"Remove stumbling repetition: '{spoken}'. Tightens pacing into the following clause.",
                        approved=False,
                    ))
                    cut_counter += 1
                    break

    return cuts

def build_edit_decision_list(
    candidate_id: str,
    start_time: float,
    end_time: float,
    proposed_cuts: List[EditDecision],
) -> EditDecisionList:
    """Consolidates proposed cuts into a chronological, non-overlapping Edit Decision List (EDL)
    and computes the resulting sequence of kept segments on the new timeline.
    """
    orig_duration = round(end_time - start_time, 3)
    if orig_duration <= 0:
        raise ValueError(f"Invalid duration: {orig_duration}s")
        
    # Filter cuts strictly within the clip boundaries
    valid_cuts = [
        c for c in proposed_cuts
        if c.original_start >= start_time and c.original_end <= end_time and c.duration_cut > 0.02
    ]
    
    # Sort cuts chronologically
    valid_cuts.sort(key=lambda c: c.original_start)
    
    # Merge overlapping cuts if any
    merged_cuts: List[EditDecision] = []
    for cut in valid_cuts:
        if not merged_cuts:
            merged_cuts.append(cut)
        else:
            prev = merged_cuts[-1]
            if cut.original_start < prev.original_end:
                # Merge
                new_end = max(prev.original_end, cut.original_end)
                new_dur = round(new_end - prev.original_start, 3)
                spoken = f"{prev.spoken_content or ''} {cut.spoken_content or ''}".strip() or None
                is_meaningful = prev.is_meaningful_content or cut.is_meaningful_content
                prev.original_end = new_end
                prev.duration_cut = new_dur
                prev.spoken_content = spoken
                prev.is_meaningful_content = is_meaningful
                prev.rationale = f"{prev.rationale} + {cut.rationale}"
            else:
                merged_cuts.append(cut)
                
    # Build kept segments
    kept_segments: List[TightenedSegment] = []
    curr_src = start_time
    curr_timeline = 0.0
    seg_idx = 1
    
    for cut in merged_cuts:
        if cut.original_start > curr_src + 0.05:
            seg_dur = round(cut.original_start - curr_src, 3)
            kept_segments.append(TightenedSegment(
                segment_index=seg_idx,
                source_start=round(curr_src, 3),
                source_end=round(cut.original_start, 3),
                timeline_start=round(curr_timeline, 3),
                timeline_end=round(curr_timeline + seg_dur, 3),
                duration=seg_dur,
            ))
            curr_timeline += seg_dur
            seg_idx += 1
        curr_src = cut.original_end
        
    # Final trailing segment
    if curr_src < end_time - 0.05:
        seg_dur = round(end_time - curr_src, 3)
        kept_segments.append(TightenedSegment(
            segment_index=seg_idx,
            source_start=round(curr_src, 3),
            source_end=round(end_time, 3),
            timeline_start=round(curr_timeline, 3),
            timeline_end=round(curr_timeline + seg_dur, 3),
            duration=seg_dur,
        ))
        curr_timeline += seg_dur
        
    tightened_duration = round(curr_timeline, 3)
    duration_saved = round(orig_duration - tightened_duration, 3)
    requires_approval = any(c.is_meaningful_content for c in merged_cuts)
    
    return EditDecisionList(
        edl_id=f"edl_{candidate_id}",
        candidate_id=candidate_id,
        original_start_time=start_time,
        original_end_time=end_time,
        original_duration=orig_duration,
        tightened_duration=tightened_duration,
        duration_saved=duration_saved,
        cuts=merged_cuts,
        kept_segments=kept_segments,
        requires_human_approval=requires_approval,
        approval_status="pending" if requires_approval else "approved",
    )

def remap_subtitles_for_edl(
    transcript: TranscriptResult,
    edl: EditDecisionList,
    output_ass_path: str,
    font_name: str = "Arial",
    font_size: int = 56,
    primary_color: str = "&H00FFFFFF",      # Crisp White
    highlight_color: str = "&H0000FFFF",    # Vibrant Yellow
    outline_color: str = "&H00000000",      # Deep Black
    margin_v: int = 300,
) -> str:
    """Remaps word-level subtitles across non-contiguous jump cuts in an EDL.
    Words that fall inside cut zones are dropped; words inside kept segments have
    their start/end timestamps translated to the new continuous timeline.
    """
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
    
    remapped_words: List[Tuple[float, float, str]] = []
    
    # Collect all words and check against kept segments
    for seg in transcript.segments:
        for w in seg.words:
            # Find if this word falls within any kept segment
            for kseg in edl.kept_segments:
                if w.end > kseg.source_start and w.start < kseg.source_end:
                    # Clip word to segment boundaries
                    w_s = max(w.start, kseg.source_start)
                    w_e = min(w.end, kseg.source_end)
                    if w_e > w_s:
                        # Remap to timeline offset
                        t_start = round(kseg.timeline_start + (w_s - kseg.source_start), 3)
                        t_end = round(kseg.timeline_start + (w_e - kseg.source_start), 3)
                        remapped_words.append((t_start, t_end, w.word))
                    break

    events: List[str] = []
    chunk_size = 4
    for i in range(0, len(remapped_words), chunk_size):
        chunk = remapped_words[i : i + chunk_size]
        chunk_start = chunk[0][0]
        chunk_end = chunk[-1][1]
        
        if chunk_end <= chunk_start:
            continue
            
        for active_idx, (w_start, w_end, w_text) in enumerate(chunk):
            line_parts = []
            for idx, (_, _, text) in enumerate(chunk):
                if idx == active_idx:
                    line_parts.append(f"{{\\c{highlight_color}\\b1}}{text}{{\\c{primary_color}\\b0}}")
                else:
                    line_parts.append(text)
            dialogue_text = " ".join(line_parts)
            start_str = format_ass_timestamp(w_start)
            end_str = format_ass_timestamp(w_end)
            events.append(f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{dialogue_text}")

    content = header + "\n".join(events) + "\n"
    out_path.write_text(content, encoding="utf-8")
    logger.info(f"Generated remapped ASS subtitles for EDL '{edl.edl_id}' ({len(events)} cues) -> {out_path}")
    return str(out_path)

def render_tightened_short(
    source_video_path: str,
    edl: EditDecisionList,
    output_mp4_path: str,
    ass_subtitle_path: Optional[str] = None,
    crop_mode: str = "pad_blur",
    focal_x: Optional[float] = None,
    test_mode: bool = False,
) -> VideoMetadata:
    """Renders non-contiguous kept segments defined in an EDL into a seamless 9:16 vertical Short
    using FFmpeg filter_complex trim + concat, strictly enforcing approval gates on spoken cuts.
    """
    out_path = Path(output_mp4_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Enforce Approval Gate for Meaningful Spoken Cuts FIRST
    if edl.requires_human_approval and edl.approval_status != "approved":
        raise PermissionError(
            f"EDL '{edl.edl_id}' contains cuts of meaningful spoken content and cannot be rendered: "
            f"approval_status is '{edl.approval_status}'. Explicit human approval is required."
        )

    if not test_mode and edl.reviewed_by in ["test_harness", "test_approval"]:
        raise PermissionError(
            f"EDL '{edl.edl_id}' has test approval ('{edl.reviewed_by}'). "
            "Production rendering requires explicit human operator approval."
        )

    in_path = Path(source_video_path).resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Source video not found: {source_video_path}")

    if not edl.kept_segments:
        raise ValueError(f"EDL '{edl.edl_id}' has no kept segments to render.")

    # Detect focal point if smart crop
    if focal_x is None:
        if crop_mode == "smart_crop":
            focal_x = detect_subject_horizontal_focal(str(in_path), edl.original_start_time, edl.original_end_time)
        else:
            focal_x = 0.5

    # Build trim + concat filter graph
    # For N segments:
    # [0:v]trim=start=S0:end=E0,setpts=PTS-STARTPTS[v0];
    # [0:a]atrim=start=S0:end=E0,asetpts=PTS-STARTPTS[a0];
    # ...
    # [v0][a0][v1][a1]...concat=n=N:v=1:a=1[vcat][acat];
    
    filter_parts = []
    concat_inputs = []
    num_segs = len(edl.kept_segments)
    
    for i, seg in enumerate(edl.kept_segments):
        v_tag = f"v{i}"
        a_tag = f"a{i}"
        filter_parts.append(
            f"[0:v]trim=start={seg.source_start:.3f}:end={seg.source_end:.3f},setpts=PTS-STARTPTS[{v_tag}]"
        )
        filter_parts.append(
            f"[0:a]atrim=start={seg.source_start:.3f}:end={seg.source_end:.3f},asetpts=PTS-STARTPTS[{a_tag}]"
        )
        concat_inputs.append(f"[{v_tag}][{a_tag}]")
        
    concat_filter = f"{''.join(concat_inputs)}concat=n={num_segs}:v=1:a=1[vcat][acat]"
    filter_parts.append(concat_filter)
    
    # Now format [vcat] into 9:16 vertical Short using vertical framing rules
    base_filter, is_complex = build_vertical_filtergraph(
        crop_mode=crop_mode,
        focal_x=focal_x,
        ass_path=ass_subtitle_path,
    )

    if is_complex:
        vertical_filter = base_filter.replace("[0:v]", "[vcat]")
    else:
        vertical_filter = f"[vcat]{base_filter}[vout]"
    filter_parts.append(vertical_filter)
    
    full_filtergraph = ";\n".join(filter_parts)
    
    # FFmpeg command
    cmd = [
        FFMPEG_PATH,
        "-y",
        "-i", str(in_path),
        "-filter_complex", full_filtergraph,
        "-map", "[vout]",
        "-map", "[acat]",
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
    ]
    
    logger.info(
        f"Rendering Tightened Short '{edl.edl_id}' ({num_segs} segments, "
        f"{edl.tightened_duration:.2f}s, saved {edl.duration_saved:.2f}s) -> '{out_path.name}'"
    )
    
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if res.returncode != 0:
        logger.error(f"FFmpeg tightened render failed:\n{res.stderr[-1000:]}")
        raise RuntimeError(f"FFmpeg render failed (code {res.returncode}): {res.stderr.strip()[-500:]}")
        
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"Rendered output MP4 is missing or empty: {out_path}")
        
    metadata = probe_video(str(out_path))
    if not metadata.has_video:
        raise ValueError(f"Render validation failed: No video stream in {out_path.name}")
        
    if metadata.video_stream.width != 1080 or metadata.video_stream.height != 1920:
        raise ValueError(
            f"Render validation failed: Expected 1080x1920, got "
            f"{metadata.video_stream.width}x{metadata.video_stream.height}"
        )
        
    logger.info(
        f"Tightened Render verified: {metadata.file_name} "
        f"({metadata.video_stream.width}x{metadata.video_stream.height}, "
        f"{metadata.duration_seconds:.2f}s, {metadata.file_size_bytes} bytes)"
    )
    return metadata

def export_edl_markdown_report(edl: EditDecisionList, output_md_path: str) -> str:
    """Generates an executive Edit Decision List (EDL) markdown report for human operator review."""
    out_p = Path(output_md_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    
    pct_saved = (edl.duration_saved / edl.original_duration * 100) if edl.original_duration > 0 else 0.0
    
    lines = [
        f"# Edit Decision List (EDL) Report: `{edl.edl_id}`",
        "",
        f"- **Target Candidate**: `{edl.candidate_id}`",
        f"- **Original Range**: `{edl.original_start_time:.2f}s` to `{edl.original_end_time:.2f}s` (Duration: `{edl.original_duration:.2f}s`)",
        f"- **Tightened Duration**: `{edl.tightened_duration:.2f}s`",
        f"- **Total Duration Saved**: `{edl.duration_saved:.2f}s` (`{pct_saved:.1f}%` pacing tightening)",
        f"- **Approval Status**: `{edl.approval_status.upper()}`",
        f"- **Meaningful Spoken Cuts Requiring Approval**: `{'YES' if edl.requires_human_approval else 'NO'}`",
        "",
        "---",
        "",
        "## Proposed Edit Decisions",
        "",
        "| Cut ID | Type | Source Window | Duration Cut | Meaningful Content? | Spoken Content / Silence | Rationale | Approved? |",
        "|---|---|---|---|---|---|---|---|",
    ]
    
    for c in edl.cuts:
        content_desc = f"`\"{c.spoken_content}\"`" if c.spoken_content else "*[Silence / Micro-pause]*"
        meaningful_tag = "**YES (Spoken Words)**" if c.is_meaningful_content else "No (Silence/Pause)"
        approved_tag = "APPROVED" if c.approved else "PENDING HUMAN APPROVAL"
        lines.append(
            f"| `{c.cut_id}` | `{c.cut_type}` | `{c.original_start:.2f}s - {c.original_end:.2f}s` | "
            f"`{c.duration_cut:.2f}s` | {meaningful_tag} | {content_desc} | {c.rationale} | `{approved_tag}` |"
        )
        
    lines.extend([
        "",
        "---",
        "",
        "## Kept Segments (Tightened Output Timeline)",
        "",
        "| Segment # | Source Timecodes | Output Timeline Timecodes | Segment Duration |",
        "|---|---|---|---|",
    ])
    
    for seg in edl.kept_segments:
        lines.append(
            f"| {seg.segment_index} | `{seg.source_start:.2f}s - {seg.source_end:.2f}s` | "
            f"`{seg.timeline_start:.2f}s - {seg.timeline_end:.2f}s` | `{seg.duration:.2f}s` |"
        )
        
    lines.extend([
        "",
        "> [!IMPORTANT]",
        "> **Human Approval Gate**: Any cut that removes meaningful spoken words must be explicitly approved by a human operator before rendering.",
        "",
    ])
    
    out_p.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Exported EDL markdown report to: {out_p}")
    return str(out_p)

def update_edl_approval(
    candidates_file: str,
    candidate_id: str,
    new_status: str,
    reviewed_by: str = "human_operator",
) -> EditDecisionList:
    """Updates approval status for a candidate's EDL in candidates.json."""
    from matos_video.config import CandidatesManifest
    p = Path(candidates_file).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Candidates file not found: {candidates_file}")

    data = json.loads(p.read_text(encoding="utf-8"))
    manifest = CandidatesManifest.model_validate(data)

    target_candidate = None
    for c in manifest.candidates:
        if c.candidate_id == candidate_id:
            target_candidate = c
            break

    if not target_candidate or not target_candidate.edl:
        raise ValueError(f"Candidate '{candidate_id}' has no EDL attached in {candidates_file}")

    now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
    target_candidate.edl.approval_status = new_status
    target_candidate.edl.reviewed_by = reviewed_by
    target_candidate.edl.review_timestamp = now_str
    
    # Also update individual cuts
    for cut in target_candidate.edl.cuts:
        if new_status == "approved":
            cut.approved = True
        elif new_status == "rejected":
            cut.approved = False

    p.write_text(json.dumps(manifest.model_dump(), indent=2), encoding="utf-8")
    logger.info(f"Updated EDL for candidate '{candidate_id}' to '{new_status}' by '{reviewed_by}'")
    return target_candidate.edl
