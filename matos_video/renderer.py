"""Render orchestrator enforcing mandatory human approval and executing Shorts exports."""
import datetime
import json
from pathlib import Path
from typing import List, Optional

from matos_video.config import (
    logger,
    CandidatesManifest,
    ClipCandidate,
    Phase2Manifest,
    TranscriptResult,
    RenderedClip,
    RenderManifest,
)
from matos_video.subtitles import generate_ass_subtitles
from matos_video.editor import cut_and_render_short

def render_candidate(
    session_dir: str,
    candidate_id: str,
    crop_mode: str = "pad_blur",
    test_mode: bool = False,
) -> RenderedClip:
    """Renders a single candidate into a 9:16 vertical Short, strictly enforcing approval checks."""
    s_dir = Path(session_dir).resolve()
    candidates_file = s_dir / "candidates.json"
    transcript_file = s_dir / "transcript.json"
    p2_manifest_file = s_dir / "phase2_manifest.json"

    if not candidates_file.exists():
        raise FileNotFoundError(f"Candidates file not found: {candidates_file}")
    if not transcript_file.exists():
        raise FileNotFoundError(f"Transcript file not found: {transcript_file}")

    candidates_manifest = CandidatesManifest.model_validate_json(candidates_file.read_text(encoding="utf-8"))
    candidate: Optional[ClipCandidate] = None
    for c in candidates_manifest.candidates:
        if c.candidate_id == candidate_id:
            candidate = c
            break

    if not candidate:
        raise ValueError(f"Candidate ID '{candidate_id}' not found in {candidates_file.name}")

    # Mandatory Human Approval Enforcement
    if candidate.approval_status != "approved":
        raise PermissionError(
            f"Candidate '{candidate_id}' cannot be rendered: approval_status is '{candidate.approval_status}'. "
            "Mandatory human operator approval is required before rendering."
        )

    if not test_mode and candidate.reviewed_by in ["test_harness", "test_approval"]:
        raise PermissionError(
            f"Candidate '{candidate_id}' is marked with test approval ('{candidate.reviewed_by}'). "
            "Production rendering requires explicit human operator approval."
        )

    # Resolve source video
    source_video_path = None
    if p2_manifest_file.exists():
        p2_data = json.loads(p2_manifest_file.read_text(encoding="utf-8"))
        source_video_path = p2_data.get("video_metadata", {}).get("file_path")

    if not source_video_path or not Path(source_video_path).exists():
        # Fallback to search in test_media
        possible = list(Path("test_media").glob(f"*{s_dir.name}*.mp4"))
        if possible:
            source_video_path = str(possible[0])
        else:
            raise FileNotFoundError(f"Could not locate source video for session: {s_dir.name}")

    # Load transcript
    transcript = TranscriptResult.model_validate_json(transcript_file.read_text(encoding="utf-8"))

    # Output paths
    render_dir = s_dir / "rendered"
    render_dir.mkdir(parents=True, exist_ok=True)

    ass_path = str(render_dir / f"{candidate_id}_subtitles.ass")
    out_mp4_path = str(render_dir / f"{candidate_id}_short.mp4")

    # Adaptive vertical margin for subtitles:
    # In pad_elevated mode, center subtitles in dedicated lower-third card zone (margin_v=380)
    # In center, zoom_blur, or standard pad_blur, place safely above mobile UI (margin_v=300)
    adaptive_margin_v = 380 if crop_mode == "pad_elevated" else 300

    # 1. Generate styled ASS subtitles
    generate_ass_subtitles(
        transcript=transcript,
        start_time=candidate.start_time,
        end_time=candidate.end_time,
        output_ass_path=ass_path,
        margin_v=adaptive_margin_v,
    )

    # 2. Render Short via FFmpeg
    meta = cut_and_render_short(
        source_video_path=source_video_path,
        start_time=candidate.start_time,
        end_time=candidate.end_time,
        output_mp4_path=out_mp4_path,
        ass_subtitle_path=ass_path,
        crop_mode=crop_mode,
    )

    clip = RenderedClip(
        clip_id=f"short_{candidate_id}",
        candidate_id=candidate.candidate_id,
        start_time=candidate.start_time,
        end_time=candidate.end_time,
        duration_seconds=meta.duration_seconds,
        resolution=f"{meta.video_stream.width}x{meta.video_stream.height}",
        crop_mode=crop_mode,
        approval_status=candidate.approval_status,
        reviewed_by=candidate.reviewed_by,
        review_timestamp=candidate.review_timestamp,
        mp4_path=out_mp4_path,
        file_size_bytes=meta.file_size_bytes,
    )

    # Update render manifest
    render_manifest_file = s_dir / "render_manifest.json"
    existing_manifest = None
    if render_manifest_file.exists():
        try:
            existing_manifest = RenderManifest.model_validate_json(render_manifest_file.read_text(encoding="utf-8"))
        except Exception:
            existing_manifest = None

    if existing_manifest:
        # Replace or append
        existing_clips = [c for c in existing_manifest.rendered_clips if c.candidate_id != candidate_id]
        existing_clips.append(clip)
        new_manifest = RenderManifest(
            source_video=s_dir.name,
            rendered_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            crop_mode_default=crop_mode,
            total_rendered=len(existing_clips),
            rendered_clips=existing_clips,
        )
    else:
        new_manifest = RenderManifest(
            source_video=s_dir.name,
            rendered_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            crop_mode_default=crop_mode,
            total_rendered=1,
            rendered_clips=[clip],
        )

    render_manifest_file.write_text(json.dumps(new_manifest.model_dump(), indent=2), encoding="utf-8")
    logger.info(f"Updated render manifest: {render_manifest_file}")

    return clip

def render_tightened_candidate(
    session_dir: str,
    candidate_id: str,
    crop_mode: str = "pad_blur",
    test_mode: bool = False,
) -> RenderedClip:
    """Renders a candidate's approved Edit Decision List (EDL) into a tightened 9:16 vertical Short,
    strictly enforcing human approval on meaningful spoken cuts."""
    from matos_video.tightener import remap_subtitles_for_edl, render_tightened_short

    s_dir = Path(session_dir).resolve()
    candidates_file = s_dir / "candidates.json"
    transcript_file = s_dir / "transcript.json"
    p2_manifest_file = s_dir / "phase2_manifest.json"

    if not candidates_file.exists():
        raise FileNotFoundError(f"Candidates file not found: {candidates_file}")
    if not transcript_file.exists():
        raise FileNotFoundError(f"Transcript file not found: {transcript_file}")

    candidates_manifest = CandidatesManifest.model_validate_json(candidates_file.read_text(encoding="utf-8"))
    candidate: Optional[ClipCandidate] = None
    for c in candidates_manifest.candidates:
        if c.candidate_id == candidate_id:
            candidate = c
            break

    if not candidate:
        raise ValueError(f"Candidate ID '{candidate_id}' not found in {candidates_file.name}")

    if not candidate.edl:
        raise ValueError(f"Candidate '{candidate_id}' does not have an Edit Decision List (EDL) attached.")

    edl = candidate.edl

    # Mandatory Approval Checks
    if candidate.approval_status != "approved":
        raise PermissionError(
            f"Candidate '{candidate_id}' cannot be rendered: candidate approval_status is '{candidate.approval_status}'."
        )

    if edl.requires_human_approval and edl.approval_status != "approved":
        raise PermissionError(
            f"EDL '{edl.edl_id}' contains meaningful spoken cuts and cannot be rendered: "
            f"approval_status is '{edl.approval_status}'. Explicit human approval is required."
        )

    if not test_mode and (
        candidate.reviewed_by in ["test_harness", "test_approval"] or
        edl.reviewed_by in ["test_harness", "test_approval"]
    ):
        raise PermissionError(
            f"Candidate or EDL has test approval. Production rendering requires explicit human operator approval."
        )

    # Resolve source video
    source_video_path = None
    if p2_manifest_file.exists():
        p2_data = json.loads(p2_manifest_file.read_text(encoding="utf-8"))
        source_video_path = p2_data.get("video_metadata", {}).get("file_path")

    if not source_video_path or not Path(source_video_path).exists():
        possible = list(Path("test_media").glob(f"*{s_dir.name}*.mp4"))
        if possible:
            source_video_path = str(possible[0])
        else:
            raise FileNotFoundError(f"Could not locate source video for session: {s_dir.name}")

    # Load transcript
    transcript = TranscriptResult.model_validate_json(transcript_file.read_text(encoding="utf-8"))

    # Output paths
    render_dir = s_dir / "rendered"
    render_dir.mkdir(parents=True, exist_ok=True)

    ass_path = str(render_dir / f"{candidate_id}_tightened_subtitles.ass")
    out_mp4_path = str(render_dir / f"{candidate_id}_tightened_short.mp4")

    adaptive_margin_v = 380 if crop_mode == "pad_elevated" else 300

    # 1. Remap ASS subtitles for the tightened timeline
    remap_subtitles_for_edl(
        transcript=transcript,
        edl=edl,
        output_ass_path=ass_path,
        margin_v=adaptive_margin_v,
    )

    # 2. Render multi-segment video using filter_complex
    meta = render_tightened_short(
        source_video_path=source_video_path,
        edl=edl,
        output_mp4_path=out_mp4_path,
        ass_subtitle_path=ass_path,
        crop_mode=crop_mode,
        test_mode=test_mode,
    )

    clip = RenderedClip(
        clip_id=f"short_{candidate_id}_tightened",
        candidate_id=candidate.candidate_id,
        start_time=candidate.start_time,
        end_time=candidate.end_time,
        duration_seconds=meta.duration_seconds,
        resolution=f"{meta.video_stream.width}x{meta.video_stream.height}",
        crop_mode=crop_mode,
        approval_status=edl.approval_status,
        reviewed_by=edl.reviewed_by,
        review_timestamp=edl.review_timestamp,
        mp4_path=out_mp4_path,
        file_size_bytes=meta.file_size_bytes,
    )

    # Update render manifest
    render_manifest_file = s_dir / "render_manifest.json"
    existing_manifest = None
    if render_manifest_file.exists():
        try:
            existing_manifest = RenderManifest.model_validate_json(render_manifest_file.read_text(encoding="utf-8"))
        except Exception:
            existing_manifest = None

    if existing_manifest:
        existing_clips = [c for c in existing_manifest.rendered_clips if c.clip_id != clip.clip_id]
        existing_clips.append(clip)
        new_manifest = RenderManifest(
            source_video=s_dir.name,
            rendered_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            crop_mode_default=crop_mode,
            total_rendered=len(existing_clips),
            rendered_clips=existing_clips,
        )
    else:
        new_manifest = RenderManifest(
            source_video=s_dir.name,
            rendered_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            crop_mode_default=crop_mode,
            total_rendered=1,
            rendered_clips=[clip],
        )

    render_manifest_file.write_text(json.dumps(new_manifest.model_dump(), indent=2), encoding="utf-8")
    logger.info(f"Updated render manifest with tightened clip: {render_manifest_file}")

    return clip

def render_all_approved(
    session_dir: str,
    crop_mode: str = "pad_blur",
    test_mode: bool = False,
) -> RenderManifest:
    """Renders all approved candidates in candidates.json."""
    s_dir = Path(session_dir).resolve()
    candidates_file = s_dir / "candidates.json"

    if not candidates_file.exists():
        raise FileNotFoundError(f"Candidates file not found: {candidates_file}")

    candidates_manifest = CandidatesManifest.model_validate_json(candidates_file.read_text(encoding="utf-8"))
    approved_candidates = [c for c in candidates_manifest.candidates if c.approval_status == "approved"]

    if not approved_candidates:
        logger.warning("No candidates currently have 'approved' status. Rendering aborted.")
        return RenderManifest(
            source_video=s_dir.name,
            rendered_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            crop_mode_default=crop_mode,
            total_rendered=0,
            rendered_clips=[],
        )

    rendered_clips = []
    for cand in approved_candidates:
        clip = render_candidate(
            session_dir=session_dir,
            candidate_id=cand.candidate_id,
            crop_mode=crop_mode,
            test_mode=test_mode,
        )
        rendered_clips.append(clip)

    manifest = RenderManifest(
        source_video=s_dir.name,
        rendered_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        crop_mode_default=crop_mode,
        total_rendered=len(rendered_clips),
        rendered_clips=rendered_clips,
    )
    return manifest
