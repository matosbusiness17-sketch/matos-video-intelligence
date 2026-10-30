"""Automated tests for AI analysis, candidate selection, edge cases, and multimodal evidence."""
import json
import pytest
from pathlib import Path

from matos_video.config import (
    TranscriptResult,
    TranscriptSegment,
    FrameInfo,
    CandidatesManifest,
)
from matos_video.analyzer import (
    find_candidate_windows,
    update_candidate_approval,
    score_hook,
    score_coherence,
)
from matos_video.gemini_connector import (
    is_headless_api_available,
    analyze_frames_headless_api,
)

SAMPLE_SESSION = Path("outputs/matos_sample_talk")

def test_score_hook_and_coherence():
    seg1 = TranscriptSegment(id=0, start=0.0, end=3.0, text="Notice the critical increase in retention!")
    hook = score_hook(seg1)
    assert hook > 0.6, "Strong hook keywords and punctuation should score high"

    seg2 = TranscriptSegment(id=1, start=3.0, end=6.0, text="This is a test,")
    coherence = score_coherence([seg1, seg2])
    assert coherence < 0.6, "Ending on comma should lower coherence score"

    seg3 = TranscriptSegment(id=2, start=6.0, end=9.0, text="And this concludes the complete thought.")
    coherence2 = score_coherence([seg1, seg3])
    assert coherence2 > 0.7, "Ending on period should increase coherence score"

def test_editorial_scoring_penalties():
    """Verifies that confusing enumeration, slow preambles, and abrupt endings are penalized."""
    # 1. Confusing enumeration penalty
    enum_seg = TranscriptSegment(id=10, start=0.0, end=4.0, text="The first was that I had a solid salary here.")
    enum_hook = score_hook(enum_seg)
    assert enum_hook < 0.35, f"Confusing enumeration should be penalized, got {enum_hook}"

    # 2. Slow preamble penalty
    preamble_seg = TranscriptSegment(id=11, start=0.0, end=5.0, text="A lot of people use the concept of what if like,")
    preamble_hook = score_hook(preamble_seg)
    assert preamble_hook < 0.40, f"Slow preamble should be penalized, got {preamble_hook}"

    # 3. Abrupt ending on conjunction/preposition
    incomplete_seg = TranscriptSegment(id=12, start=5.0, end=8.0, text="If you don't have the discipline")
    abrupt_coherence = score_coherence([incomplete_seg])
    assert abrupt_coherence < 0.50, f"Incomplete thought / abrupt ending should be penalized, got {abrupt_coherence}"

    # 4. Clean punchy hook without preamble
    punchy_seg = TranscriptSegment(id=13, start=0.0, end=3.0, text="What if it all goes wrong?")
    punchy_hook = score_hook(punchy_seg)
    assert punchy_hook >= 0.65, f"Punchy question hook should score high, got {punchy_hook}"

def test_candidate_selection_standard():
    transcript_path = SAMPLE_SESSION / "transcript.json"
    frames_path = SAMPLE_SESSION / "frames" / "frames_manifest.json"
    assert transcript_path.exists()
    assert frames_path.exists()

    t_data = json.loads(transcript_path.read_text(encoding="utf-8"))
    f_data = json.loads(frames_path.read_text(encoding="utf-8"))

    transcript = TranscriptResult.model_validate(t_data)
    frames = [FrameInfo.model_validate(f) for f in f_data]

    candidates = find_candidate_windows(
        transcript=transcript,
        frames=frames,
        min_duration=15.0,
        max_duration=35.0,
        target_duration=25.0,
    )

    assert len(candidates) >= 1
    c1 = candidates[0]
    assert 15.0 <= c1.duration <= 35.0
    assert c1.start_time >= 0.0
    assert c1.end_time > c1.start_time
    assert len(c1.observed_visual_evidence.keyframe_timestamps) > 0
    assert c1.approval_status == "pending"

def test_edge_case_empty_transcript():
    """Edge case: Silent video with zero transcript segments."""
    empty_transcript = TranscriptResult(
        language="en",
        duration_seconds=30.0,
        device_used="cpu",
        compute_type="int8",
        model_name="tiny",
        segments=[],
    )
    dummy_frames = [
        FrameInfo(frame_index=0, timestamp_seconds=0.0, file_path="f0.jpg", relative_path="f0.jpg", width=1280, height=720),
        FrameInfo(frame_index=30, timestamp_seconds=1.0, file_path="f1.jpg", relative_path="f1.jpg", width=1280, height=720),
    ]

    candidates = find_candidate_windows(empty_transcript, dummy_frames, min_duration=10.0, max_duration=30.0)
    assert len(candidates) == 1
    assert "no dialogue detected" in candidates[0].headline.lower()
    assert candidates[0].uncertainty_notes is not None

def test_edge_case_short_video():
    """Edge case: Video total duration is shorter than min_duration."""
    short_transcript = TranscriptResult(
        language="en",
        duration_seconds=8.0,
        device_used="cpu",
        compute_type="int8",
        model_name="tiny",
        segments=[
            TranscriptSegment(id=0, start=0.0, end=4.0, text="A brief intro."),
            TranscriptSegment(id=1, start=4.0, end=8.0, text="And a quick conclusion."),
        ],
    )
    candidates = find_candidate_windows(short_transcript, [], min_duration=25.0, max_duration=45.0)
    assert len(candidates) == 1
    assert candidates[0].duration == 8.0
    assert "shorter than target minimum" in candidates[0].uncertainty_notes

def test_referenced_keyframes_exist_on_disk():
    """Verifies that all frames referenced in candidates.json exist as real files."""
    candidates_path = SAMPLE_SESSION / "candidates.json"
    assert candidates_path.exists()

    data = json.loads(candidates_path.read_text(encoding="utf-8"))
    manifest = CandidatesManifest.model_validate(data)

    frames_dir = SAMPLE_SESSION / "frames"
    for c in manifest.candidates:
        assert len(c.observed_visual_evidence.keyframe_paths) > 0
        for rel_path in c.observed_visual_evidence.keyframe_paths:
            full_frame_path = frames_dir / rel_path
            assert full_frame_path.exists(), f"Referenced frame missing: {full_frame_path}"

def test_multimodal_unspoken_facts_demonstrated():
    """Confirms at least three distinct visual facts not present in the transcript are cataloged."""
    candidates_path = SAMPLE_SESSION / "candidates.json"
    data = json.loads(candidates_path.read_text(encoding="utf-8"))
    manifest = CandidatesManifest.model_validate(data)

    all_unspoken = []
    for c in manifest.candidates:
        all_unspoken.extend(c.observed_visual_evidence.unspoken_visual_facts)

    assert len(all_unspoken) >= 3, f"Expected at least 3 unspoken facts, found: {len(all_unspoken)}"
    # Verify specific factual details
    assert any("ENG-942" in f for f in all_unspoken)
    assert any("82.4%" in f for f in all_unspoken)
    assert any("9:16" in f or "RTX 5070" in f for f in all_unspoken)

def test_human_approval_workflow(tmp_path):
    """Tests approval status mutation and audit trail."""
    test_json = tmp_path / "candidates.json"
    orig_data = json.loads((SAMPLE_SESSION / "candidates.json").read_text(encoding="utf-8"))
    test_json.write_text(json.dumps(orig_data), encoding="utf-8")

    # Approve candidate_02
    manifest = update_candidate_approval(
        str(test_json),
        candidate_id="candidate_02",
        new_status="approved",
        reviewed_by="Lead Automation Engineer",
    )
    c2 = next(c for c in manifest.candidates if c.candidate_id == "candidate_02")
    assert c2.approval_status == "approved"
    assert c2.reviewed_by == "Lead Automation Engineer"
    assert c2.review_timestamp is not None

def test_headless_api_fallback_behavior():
    """Tests that headless API safely reports unavailable without crashing when no key is set."""
    res = analyze_frames_headless_api([], "test transcript")
    assert isinstance(res, dict)
    assert "available" in res
    if not is_headless_api_available():
        assert res["available"] is False
        assert "GEMINI_API_KEY is not configured" in res["reason"]
