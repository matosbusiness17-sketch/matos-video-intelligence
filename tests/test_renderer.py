"""Automated tests for renderer approval gating, test-mode separation, and render manifest output."""
import json
import pytest
from pathlib import Path

from matos_video.config import CandidatesManifest, ClipCandidate, ObservedVisualEvidence, SelectionRationale
from matos_video.renderer import render_candidate, render_all_approved

SAMPLE_SESSION = Path("outputs/matos_podcast_demo")

def test_render_blocks_pending_candidate(tmp_path):
    """Ensures rendering is strictly blocked for candidates in pending status."""
    test_session = tmp_path / "session"
    test_session.mkdir()

    # Create dummy candidates.json with pending candidate
    manifest = CandidatesManifest(
        source_video="test",
        analysis_mode="interactive_antigravity",
        min_duration=20.0,
        max_duration=45.0,
        target_duration=30.0,
        total_candidates_found=1,
        candidates=[
            ClipCandidate(
                candidate_id="cand_pending",
                start_time=0.0,
                end_time=25.0,
                duration=25.0,
                headline="Pending Clip",
                hook_summary="Hook",
                transcript_excerpt="Excerpt",
                transcript_segment_ids=[],
                observed_visual_evidence=ObservedVisualEvidence(),
                selection_rationale=SelectionRationale(hook_score=0.8, flow_coherence=0.8, visual_engagement=0.8, overall_score=0.8, justification=""),
                confidence_score=0.8,
                approval_status="pending",
            )
        ]
    )
    (test_session / "candidates.json").write_text(json.dumps(manifest.model_dump()), encoding="utf-8")
    (test_session / "transcript.json").write_text(json.dumps({"language": "en", "duration_seconds": 30.0, "device_used": "cpu", "compute_type": "int8", "model_name": "tiny", "segments": []}), encoding="utf-8")

    with pytest.raises(PermissionError) as exc_info:
        render_candidate(str(test_session), "cand_pending")
    assert "approval_status is 'pending'" in str(exc_info.value)
    assert "Mandatory human operator approval" in str(exc_info.value)

def test_render_blocks_rejected_candidate(tmp_path):
    """Ensures rendering is strictly blocked for rejected candidates."""
    test_session = tmp_path / "session"
    test_session.mkdir()

    manifest = CandidatesManifest(
        source_video="test",
        analysis_mode="interactive_antigravity",
        min_duration=20.0,
        max_duration=45.0,
        target_duration=30.0,
        total_candidates_found=1,
        candidates=[
            ClipCandidate(
                candidate_id="cand_rejected",
                start_time=0.0,
                end_time=25.0,
                duration=25.0,
                headline="Rejected Clip",
                hook_summary="Hook",
                transcript_excerpt="Excerpt",
                transcript_segment_ids=[],
                observed_visual_evidence=ObservedVisualEvidence(),
                selection_rationale=SelectionRationale(hook_score=0.8, flow_coherence=0.8, visual_engagement=0.8, overall_score=0.8, justification=""),
                confidence_score=0.8,
                approval_status="rejected",
            )
        ]
    )
    (test_session / "candidates.json").write_text(json.dumps(manifest.model_dump()), encoding="utf-8")
    (test_session / "transcript.json").write_text(json.dumps({"language": "en", "duration_seconds": 30.0, "device_used": "cpu", "compute_type": "int8", "model_name": "tiny", "segments": []}), encoding="utf-8")

    with pytest.raises(PermissionError) as exc_info:
        render_candidate(str(test_session), "cand_rejected")
    assert "approval_status is 'rejected'" in str(exc_info.value)

def test_render_blocks_test_harness_in_production_mode(tmp_path):
    """Ensures test_harness approvals cannot be rendered without explicit test_mode flag."""
    test_session = tmp_path / "session"
    test_session.mkdir()

    manifest = CandidatesManifest(
        source_video="test",
        analysis_mode="interactive_antigravity",
        min_duration=20.0,
        max_duration=45.0,
        target_duration=30.0,
        total_candidates_found=1,
        candidates=[
            ClipCandidate(
                candidate_id="cand_test",
                start_time=0.0,
                end_time=25.0,
                duration=25.0,
                headline="Test Clip",
                hook_summary="Hook",
                transcript_excerpt="Excerpt",
                transcript_segment_ids=[],
                observed_visual_evidence=ObservedVisualEvidence(),
                selection_rationale=SelectionRationale(hook_score=0.8, flow_coherence=0.8, visual_engagement=0.8, overall_score=0.8, justification=""),
                confidence_score=0.8,
                approval_status="approved",
                reviewed_by="test_harness",
            )
        ]
    )
    (test_session / "candidates.json").write_text(json.dumps(manifest.model_dump()), encoding="utf-8")
    (test_session / "transcript.json").write_text(json.dumps({"language": "en", "duration_seconds": 30.0, "device_used": "cpu", "compute_type": "int8", "model_name": "tiny", "segments": []}), encoding="utf-8")

    with pytest.raises(PermissionError) as exc_info:
        render_candidate(str(test_session), "cand_test", test_mode=False)
    assert "marked with test approval" in str(exc_info.value)

def test_render_manifest_generation():
    """Verifies that render_manifest.json in outputs/matos_podcast_demo is valid and complete."""
    manifest_file = SAMPLE_SESSION / "render_manifest.json"
    assert manifest_file.exists()
    data = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert data["source_video"] == "matos_podcast_demo"
    assert data["total_rendered"] >= 2
    for clip in data["rendered_clips"]:
        assert clip["resolution"] == "1080x1920"
        assert clip["duration_seconds"] > 20.0
        assert Path(clip["mp4_path"]).exists()
        assert clip["file_size_bytes"] > 500000
