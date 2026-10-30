"""Command-line interface for Matos Video Intelligence (n8n-ready)."""
import argparse
import json
import sys
from pathlib import Path

from matos_video.config import (
    WORKSPACE_DIR,
    OUTPUTS_DIR,
    Phase2Manifest,
    logger,
)
from matos_video.intake import probe_video, download_video
from matos_video.transcribe import (
    extract_audio,
    transcribe_audio,
    export_transcript_json,
    export_transcript_srt,
)
from matos_video.frames import extract_frames, extract_frame_at_timestamp

def cmd_probe(args):
    metadata = probe_video(args.input)
    if args.json:
        print(json.dumps(metadata.model_dump(), indent=2))
    else:
        print(f"File: {metadata.file_name}")
        print(f"Duration: {metadata.duration_seconds}s")
        if metadata.video_stream:
            print(f"Video: {metadata.video_stream.width}x{metadata.video_stream.height} @ {metadata.video_stream.fps} fps ({metadata.video_stream.codec_name})")
        if metadata.audio_stream:
            print(f"Audio: {metadata.audio_stream.sample_rate} Hz, {metadata.audio_stream.channels} ch ({metadata.audio_stream.codec_name})")

def cmd_download(args):
    out_dir = args.output_dir or str(OUTPUTS_DIR / "downloads")
    downloaded = download_video(args.url, out_dir)
    if args.json:
        print(json.dumps({"downloaded_file": downloaded}))
    else:
        print(f"Downloaded: {downloaded}")

def cmd_transcribe(args):
    in_path = Path(args.input).resolve()
    out_dir = Path(args.output_dir or str(OUTPUTS_DIR / in_path.stem)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # If input is video, extract audio first
    if in_path.suffix.lower() in [".mp4", ".mov", ".mkv", ".webm", ".avi"]:
        wav_path = str(out_dir / "audio.wav")
        extract_audio(str(in_path), wav_path)
        audio_to_transcribe = wav_path
    else:
        audio_to_transcribe = str(in_path)

    transcript = transcribe_audio(audio_to_transcribe, model_size=args.model, language=args.language)
    json_path = export_transcript_json(transcript, str(out_dir / "transcript.json"))
    srt_path = export_transcript_srt(transcript, str(out_dir / "transcript.srt"))

    if args.json:
        print(json.dumps({
            "transcript_json": json_path,
            "transcript_srt": srt_path,
            "language": transcript.language,
            "duration": transcript.duration_seconds,
            "segments_count": len(transcript.segments),
            "device_used": transcript.device_used,
            "compute_type": transcript.compute_type,
        }, indent=2))
    else:
        print(f"Transcribed {transcript.duration_seconds}s in {transcript.language}")
        print(f"JSON: {json_path}")
        print(f"SRT:  {srt_path}")

def cmd_extract_frames(args):
    in_path = Path(args.input).resolve()
    out_dir = Path(args.output_dir or str(OUTPUTS_DIR / in_path.stem / "frames")).resolve()
    frames = extract_frames(
        str(in_path),
        str(out_dir),
        interval_seconds=args.interval,
        scene_threshold=args.threshold,
    )
    if args.json:
        print(json.dumps({
            "output_dir": str(out_dir),
            "total_frames": len(frames),
            "scene_changes": sum(1 for f in frames if f.is_scene_change),
        }, indent=2))
    else:
        print(f"Extracted {len(frames)} frames into {out_dir}")

def cmd_pipeline(args):
    in_path = Path(args.input).resolve()
    base_name = in_path.stem
    session_dir = Path(args.output_dir or str(OUTPUTS_DIR / base_name)).resolve()
    session_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== Starting Phase 2 Pipeline for '{in_path.name}' ===")

    # 1. Probe video
    metadata = probe_video(str(in_path))

    # 2. Extract audio
    wav_path = str(session_dir / "audio.wav")
    extract_audio(str(in_path), wav_path)

    # 3. Transcribe audio
    transcript = transcribe_audio(wav_path, model_size=args.model, language=args.language)
    export_transcript_json(transcript, str(session_dir / "transcript.json"))
    export_transcript_srt(transcript, str(session_dir / "transcript.srt"))

    # 4. Extract intelligent frames
    frames_dir = session_dir / "frames"
    frames = extract_frames(
        str(in_path),
        str(frames_dir),
        interval_seconds=args.interval,
        scene_threshold=args.threshold,
    )

    manifest = Phase2Manifest(
        video_metadata=metadata,
        transcript=transcript,
        audio_wav_path=wav_path,
        frames_directory=str(frames_dir),
        total_frames_extracted=len(frames),
        frames=frames,
    )

    manifest_file = session_dir / "phase2_manifest.json"
    manifest_file.write_text(json.dumps(manifest.model_dump(), indent=2), encoding="utf-8")
    logger.info(f"Phase 2 manifest saved to: {manifest_file}")

    if args.json:
        print(json.dumps(manifest.model_dump(), indent=2))
    else:
        print("\n=== Phase 2 Pipeline Execution Complete ===")
        print(f"Video: {metadata.file_name} ({metadata.duration_seconds}s)")
        print(f"Audio: {metadata.audio_stream.sample_rate if metadata.audio_stream else 0}Hz WAV extracted")
        print(f"Transcription: {len(transcript.segments)} segments, device={transcript.device_used} ({transcript.compute_type})")
        print(f"Frames: {len(frames)} extracted ({sum(1 for f in frames if f.is_scene_change)} scene changes)")
        print(f"Session directory: {session_dir}")

def cmd_analyze(args):
    from matos_video.config import TranscriptResult, FrameInfo, CandidatesManifest
    from matos_video.analyzer import (
        find_candidate_windows,
        export_candidates_json,
        export_analysis_markdown_report,
    )

    session_dir = Path(args.session_dir).resolve()
    transcript_path = session_dir / "transcript.json"
    frames_manifest_path = session_dir / "frames" / "frames_manifest.json"

    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript not found at {transcript_path}")

    transcript_data = json.loads(transcript_path.read_text(encoding="utf-8"))
    transcript = TranscriptResult.model_validate(transcript_data)

    frames: List[FrameInfo] = []
    if frames_manifest_path.exists():
        frames_data = json.loads(frames_manifest_path.read_text(encoding="utf-8"))
        frames = [FrameInfo.model_validate(f) for f in frames_data]

    candidates = find_candidate_windows(
        transcript=transcript,
        frames=frames,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        target_duration=args.target_duration,
    )

    manifest = CandidatesManifest(
        source_video=session_dir.name,
        analysis_mode=args.mode,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        target_duration=args.target_duration,
        total_candidates_found=len(candidates),
        candidates=candidates,
    )

    json_out = export_candidates_json(manifest, str(session_dir / "candidates.json"))
    md_out = export_analysis_markdown_report(manifest, str(session_dir / "analysis_report.md"))

    if args.json:
        print(json.dumps(manifest.model_dump(), indent=2))
    else:
        print(f"Analysis complete: {len(candidates)} candidate(s) identified.")
        print(f"Candidates JSON: {json_out}")
        print(f"Analysis Report: {md_out}")

def cmd_approve(args):
    from matos_video.analyzer import update_candidate_approval
    from matos_video.tightener import update_edl_approval

    if args.edl:
        edl = update_edl_approval(
            candidates_file=args.candidates,
            candidate_id=args.id,
            new_status=args.status,
            reviewed_by=args.by or "human_operator",
        )
        if args.json:
            print(json.dumps(edl.model_dump(), indent=2))
        else:
            print(f"Updated EDL for candidate '{args.id}' to '{args.status}' by '{args.by or 'human_operator'}'")
    else:
        manifest = update_candidate_approval(
            candidates_file=args.candidates,
            candidate_id=args.id,
            new_status=args.status,
            reviewed_by=args.by or "human_operator",
        )
        if args.json:
            print(json.dumps(manifest.model_dump(), indent=2))
        else:
            print(f"Updated candidate '{args.id}' to '{args.status}' by '{args.by or 'human_operator'}'")

def cmd_tighten(args):
    from matos_video.config import CandidatesManifest, TranscriptResult
    from matos_video.tightener import (
        detect_silence_cuts,
        detect_filler_and_repetition,
        build_edit_decision_list,
        export_edl_markdown_report,
    )

    s_dir = Path(args.session_dir).resolve()
    cand_path = s_dir / "candidates.json"
    transcript_path = s_dir / "transcript.json"

    if not cand_path.exists():
        raise FileNotFoundError(f"Candidates file not found: {cand_path}")
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript file not found: {transcript_path}")

    transcript = TranscriptResult.model_validate_json(transcript_path.read_text(encoding="utf-8"))
    manifest = CandidatesManifest.model_validate_json(cand_path.read_text(encoding="utf-8"))

    target_candidate = None
    for c in manifest.candidates:
        if c.candidate_id == args.candidate_id:
            target_candidate = c
            break

    if not target_candidate:
        raise ValueError(f"Candidate '{args.candidate_id}' not found in manifest.")

    # 1. Detect silence pauses
    silence_cuts = detect_silence_cuts(
        transcript=transcript,
        start_time=target_candidate.start_time,
        end_time=target_candidate.end_time,
        min_pause_duration=args.min_pause,
        retain_pause=args.retain_pause,
    )

    # 2. Detect repetition and filler
    repetition_cuts = detect_filler_and_repetition(
        transcript=transcript,
        start_time=target_candidate.start_time,
        end_time=target_candidate.end_time,
    )

    all_cuts = silence_cuts + repetition_cuts
    edl = build_edit_decision_list(
        candidate_id=target_candidate.candidate_id,
        start_time=target_candidate.start_time,
        end_time=target_candidate.end_time,
        proposed_cuts=all_cuts,
    )

    # Attach EDL to candidate
    target_candidate.edl = edl
    target_candidate.tightened_duration = edl.tightened_duration

    # Export manifest
    cand_path.write_text(json.dumps(manifest.model_dump(), indent=2), encoding="utf-8")

    # Export EDL report
    edl_md_path = s_dir / f"edl_{target_candidate.candidate_id}.md"
    export_edl_markdown_report(edl, str(edl_md_path))

    if args.json:
        print(json.dumps(edl.model_dump(), indent=2))
    else:
        print(f"Smart Tightening EDL generated for '{target_candidate.candidate_id}':")
        print(f"Original duration:  {edl.original_duration:.2f}s")
        print(f"Tightened duration: {edl.tightened_duration:.2f}s (saved {edl.duration_saved:.2f}s)")
        print(f"Proposed cuts:      {len(edl.cuts)} ({sum(1 for c in edl.cuts if c.is_meaningful_content)} spoken content cuts)")
        print(f"Approval required:  {'YES' if edl.requires_human_approval else 'NO'}")
        print(f"EDL Markdown:       {edl_md_path}")

def cmd_render(args):
    from matos_video.renderer import render_candidate, render_tightened_candidate, render_all_approved

    if args.tightened:
        if not args.candidate_id:
            print("Error: --tightened requires --candidate-id <id>")
            sys.exit(1)
        clip = render_tightened_candidate(
            session_dir=args.session_dir,
            candidate_id=args.candidate_id,
            crop_mode=args.crop_mode,
            test_mode=args.test_mode,
        )
        if args.json:
            print(json.dumps(clip.model_dump(), indent=2))
        else:
            print(f"Rendered Tightened Short successfully: {clip.clip_id}")
            print(f"Output: {clip.mp4_path} ({clip.resolution}, {clip.duration_seconds:.2f}s)")
    elif args.candidate_id:
        clip = render_candidate(
            session_dir=args.session_dir,
            candidate_id=args.candidate_id,
            crop_mode=args.crop_mode,
            test_mode=args.test_mode,
        )
        if args.json:
            print(json.dumps(clip.model_dump(), indent=2))
        else:
            print(f"Rendered Short successfully: {clip.clip_id}")
            print(f"Output: {clip.mp4_path} ({clip.resolution}, {clip.duration_seconds:.2f}s)")
    elif args.all_approved:
        manifest = render_all_approved(
            session_dir=args.session_dir,
            crop_mode=args.crop_mode,
            test_mode=args.test_mode,
        )
        if args.json:
            print(json.dumps(manifest.model_dump(), indent=2))
        else:
            print(f"Rendered {manifest.total_rendered} approved clip(s).")
            for c in manifest.rendered_clips:
                print(f"- {c.clip_id}: {c.mp4_path}")
    else:
        print("Please specify either --candidate-id <id> or --all-approved")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Matos Video Intelligence CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # probe
    p_probe = subparsers.add_parser("probe", help="Probe video metadata")
    p_probe.add_argument("--input", "-i", required=True, help="Path to video file")
    p_probe.add_argument("--json", action="store_true", help="Output as JSON")
    p_probe.set_defaults(func=cmd_probe)

    # download
    p_dl = subparsers.add_parser("download", help="Download video via yt-dlp")
    p_dl.add_argument("--url", "-u", required=True, help="Video URL")
    p_dl.add_argument("--output-dir", "-o", help="Target output directory")
    p_dl.add_argument("--json", action="store_true", help="Output as JSON")
    p_dl.set_defaults(func=cmd_download)

    # transcribe
    p_tr = subparsers.add_parser("transcribe", help="Transcribe audio or video")
    p_tr.add_argument("--input", "-i", required=True, help="Path to video or audio file")
    p_tr.add_argument("--output-dir", "-o", help="Target output directory")
    p_tr.add_argument("--model", "-m", default="tiny", help="Whisper model size (tiny, base, small)")
    p_tr.add_argument("--language", "-l", help="Language code (e.g. en, fr, de)")
    p_tr.add_argument("--json", action="store_true", help="Output as JSON")
    p_tr.set_defaults(func=cmd_transcribe)

    # extract-frames
    p_fr = subparsers.add_parser("extract-frames", help="Extract intelligent frames")
    p_fr.add_argument("--input", "-i", required=True, help="Path to video file")
    p_fr.add_argument("--output-dir", "-o", help="Target output directory")
    p_fr.add_argument("--interval", type=float, default=2.0, help="Interval in seconds")
    p_fr.add_argument("--threshold", type=float, default=0.03, help="Scene change threshold")
    p_fr.add_argument("--json", action="store_true", help="Output as JSON")
    p_fr.set_defaults(func=cmd_extract_frames)

    # pipeline
    p_pipe = subparsers.add_parser("pipeline", help="Run full Phase 2 pipeline")
    p_pipe.add_argument("--input", "-i", required=True, help="Path to video file")
    p_pipe.add_argument("--output-dir", "-o", help="Target session directory")
    p_pipe.add_argument("--model", "-m", default="tiny", help="Whisper model size")
    p_pipe.add_argument("--language", "-l", help="Language code")
    p_pipe.add_argument("--interval", type=float, default=2.0, help="Interval in seconds")
    p_pipe.add_argument("--threshold", type=float, default=0.03, help="Scene change threshold")
    p_pipe.add_argument("--json", action="store_true", help="Output as JSON")
    p_pipe.set_defaults(func=cmd_pipeline)

    # analyze (Phase 3)
    p_an = subparsers.add_parser("analyze", help="Analyze transcript and frames to select highlight candidates")
    p_an.add_argument("--session-dir", "-s", required=True, help="Path to session directory containing transcript.json and frames/")
    p_an.add_argument("--min-duration", type=float, default=20.0, help="Minimum clip duration in seconds")
    p_an.add_argument("--max-duration", type=float, default=45.0, help="Maximum clip duration in seconds")
    p_an.add_argument("--target-duration", type=float, default=30.0, help="Ideal target duration in seconds")
    p_an.add_argument("--mode", default="interactive_antigravity", help="Analysis mode (interactive_antigravity or headless_gemini_api)")
    p_an.add_argument("--json", action="store_true", help="Output as JSON")
    p_an.set_defaults(func=cmd_analyze)

    # approve (Phase 3 & 5)
    p_app = subparsers.add_parser("approve", help="Update human approval status for a candidate or EDL")
    p_app.add_argument("--candidates", "-c", required=True, help="Path to candidates.json file")
    p_app.add_argument("--id", required=True, help="Candidate ID (e.g. candidate_01)")
    p_app.add_argument("--status", choices=["approved", "rejected", "pending"], required=True, help="New approval status")
    p_app.add_argument("--by", default="human_operator", help="Reviewer identifier")
    p_app.add_argument("--edl", action="store_true", help="Approve EDL rather than raw candidate")
    p_app.add_argument("--json", action="store_true", help="Output as JSON")
    p_app.set_defaults(func=cmd_approve)

    # tighten (Phase 5)
    p_tight = subparsers.add_parser("tighten", help="Generate Smart Tightening EDL for a candidate")
    p_tight.add_argument("--session-dir", "-s", required=True, help="Session directory containing candidates.json and transcript.json")
    p_tight.add_argument("--candidate-id", "-c", required=True, help="Candidate ID to tighten (e.g. candidate_02)")
    p_tight.add_argument("--min-pause", type=float, default=0.50, help="Minimum silence pause duration to shorten (seconds)")
    p_tight.add_argument("--retain-pause", type=float, default=0.25, help="Natural pause duration to retain (seconds)")
    p_tight.add_argument("--json", action="store_true", help="Output as JSON")
    p_tight.set_defaults(func=cmd_tighten)

    # render (Phase 4 & 5)
    p_ren = subparsers.add_parser("render", help="Render approved candidates into vertical 9:16 Shorts")
    p_ren.add_argument("--session-dir", "-s", required=True, help="Session directory containing candidates.json")
    p_ren.add_argument("--candidate-id", "-c", help="Specific candidate ID to render")
    p_ren.add_argument("--all-approved", action="store_true", help="Render all approved candidates")
    p_ren.add_argument("--tightened", action="store_true", help="Render candidate's approved EDL")
    p_ren.add_argument(
        "--crop-mode",
        choices=["pad_blur", "pad_elevated", "zoom_blur", "center", "smart_crop"],
        default="pad_blur",
        help="Vertical framing mode: pad_blur (ambient blur), pad_elevated (upper card + lower captions), zoom_blur (1.25x scale), center (portrait talking heads only), smart_crop (edge-energy biased)",
    )
    p_ren.add_argument("--test-mode", action="store_true", help="Permit test-only approvals (for automated test harnesses)")
    p_ren.add_argument("--json", action="store_true", help="Output as JSON")
    p_ren.set_defaults(func=cmd_render)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
