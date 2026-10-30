"""Automated tests for Smart Tightening, EDL generation, subtitle remapping, and approval gates."""
import json
import pytest
from pathlib import Path

from matos_video.config import (
    TranscriptResult,
    TranscriptSegment,
    WordTimestamp,
    EditDecision,
    EditDecisionList,
    CandidatesManifest,
    ClipCandidate,
    ObservedVisualEvidence,
    SelectionRationale,
)
from matos_video.tightener import (
    detect_silence_cuts,
    detect_filler_and_repetition,
    build_edit_decision_list,
    remap_subtitles_for_edl,
    render_tightened_short,
    update_edl_approval,
)
from matos_video.renderer import render_tightened_candidate

def make_sample_transcript() -> TranscriptResult:
    """Builds a synthetic transcript with deliberate pauses and a repetitive stumble."""
    seg1 = TranscriptSegment(
        id=1,
        start=10.0,
        end=14.0,
        text="What if it all goes wrong?",
        words=[
            WordTimestamp(word="What", start=10.0, end=10.3, probability=0.99),
            WordTimestamp(word="if", start=10.3, end=10.5, probability=0.99),
            WordTimestamp(word="it", start=10.5, end=10.7, probability=0.99),
            WordTimestamp(word="all", start=10.7, end=11.0, probability=0.99),
            WordTimestamp(word="goes", start=11.0, end=11.4, probability=0.99),
            WordTimestamp(word="wrong?", start=11.4, end=11.8, probability=0.99),
            # Large dead air pause: 11.8s to 12.8s (1.0s gap)
            WordTimestamp(word="What", start=12.8, end=13.1, probability=0.99),
            WordTimestamp(word="if", start=13.1, end=13.3, probability=0.99),
            WordTimestamp(word="it", start=13.3, end=13.5, probability=0.99),
            WordTimestamp(word="works", start=13.5, end=13.8, probability=0.99),
            WordTimestamp(word="out?", start=13.8, end=14.0, probability=0.99),
        ],
    )
    seg2 = TranscriptSegment(
        id=2,
        start=14.0,
        end=20.0,
        text="get into that place of expansion and that place of expansion, what if anything is possible?",
        words=[
            WordTimestamp(word="get", start=14.0, end=14.3, probability=0.95),
            WordTimestamp(word="into", start=14.3, end=14.6, probability=0.95),
            WordTimestamp(word="that", start=14.6, end=14.8, probability=0.95),
            WordTimestamp(word="place", start=14.8, end=15.1, probability=0.95),
            WordTimestamp(word="of", start=15.1, end=15.3, probability=0.95),
            WordTimestamp(word="expansion", start=15.3, end=15.8, probability=0.95),
            # Repetitive phrase stumble: "and that place of expansion"
            WordTimestamp(word="and", start=15.8, end=16.0, probability=0.95),
            WordTimestamp(word="that", start=16.0, end=16.2, probability=0.95),
            WordTimestamp(word="place", start=16.2, end=16.5, probability=0.95),
            WordTimestamp(word="of", start=16.5, end=16.7, probability=0.95),
            WordTimestamp(word="expansion", start=16.7, end=17.2, probability=0.95),
            WordTimestamp(word="what", start=17.5, end=17.8, probability=0.95),
            WordTimestamp(word="if", start=17.8, end=18.0, probability=0.95),
            WordTimestamp(word="anything", start=18.0, end=18.5, probability=0.95),
            WordTimestamp(word="is", start=18.5, end=18.7, probability=0.95),
            WordTimestamp(word="possible?", start=18.7, end=19.5, probability=0.95),
        ],
    )
    return TranscriptResult(
        language="en",
        duration_seconds=30.0,
        device_used="cpu",
        compute_type="int8",
        model_name="base",
        segments=[seg1, seg2],
    )

def test_detect_silence_cuts():
    """Verifies that gaps exceeding min_pause_duration are detected and trimmed."""
    tr = make_sample_transcript()
    cuts = detect_silence_cuts(tr, start_time=10.0, end_time=20.0, min_pause_duration=0.5, retain_pause=0.25)
    
    assert len(cuts) >= 1
    silence_cut = cuts[0]
    assert silence_cut.cut_type == "silence_trim"
    assert silence_cut.is_meaningful_content is False
    # The gap was 11.8s to 12.8s (1.0s gap). Retaining 0.25s leaves 0.75s cut.
    assert 0.70 <= silence_cut.duration_cut <= 0.80

def test_detect_filler_and_repetition():
    """Verifies detection of repeated phrase stumbles."""
    tr = make_sample_transcript()
    cuts = detect_filler_and_repetition(tr, start_time=10.0, end_time=20.0)
    
    assert len(cuts) >= 1
    rep_cut = cuts[0]
    assert rep_cut.cut_type == "repetition_removal"
    assert rep_cut.is_meaningful_content is True
    assert rep_cut.approved is False
    assert "that place of" in rep_cut.spoken_content

def test_build_edl_timeline_math():
    """Verifies EDL calculation, kept segments, and duration arithmetic."""
    c1 = EditDecision(
        cut_id="cut_01",
        cut_type="silence_trim",
        original_start=12.0,
        original_end=13.0,
        duration_cut=1.0,
        spoken_content=None,
        is_meaningful_content=False,
        rationale="Silence trim",
        approved=True,
    )
    c2 = EditDecision(
        cut_id="cut_02",
        cut_type="repetition_removal",
        original_start=16.0,
        original_end=17.5,
        duration_cut=1.5,
        spoken_content="repetition",
        is_meaningful_content=True,
        rationale="Repetition removal",
        approved=False,
    )
    
    edl = build_edit_decision_list(
        candidate_id="cand_test",
        start_time=10.0,
        end_time=20.0,
        proposed_cuts=[c1, c2],
    )
    
    assert edl.original_duration == 10.0
    assert edl.duration_saved == 2.5
    assert edl.tightened_duration == 7.5
    assert len(edl.kept_segments) == 3
    
    # Segment 1: [10.0 -> 12.0] (2.0s) -> timeline [0.0 -> 2.0]
    assert edl.kept_segments[0].source_start == 10.0
    assert edl.kept_segments[0].source_end == 12.0
    assert edl.kept_segments[0].timeline_start == 0.0
    assert edl.kept_segments[0].timeline_end == 2.0
    
    # Segment 2: [13.0 -> 16.0] (3.0s) -> timeline [2.0 -> 5.0]
    assert edl.kept_segments[1].source_start == 13.0
    assert edl.kept_segments[1].source_end == 16.0
    assert edl.kept_segments[1].timeline_start == 2.0
    assert edl.kept_segments[1].timeline_end == 5.0
    
    # Segment 3: [17.5 -> 20.0] (2.5s) -> timeline [5.0 -> 7.5]
    assert edl.kept_segments[2].source_start == 17.5
    assert edl.kept_segments[2].source_end == 20.0
    assert edl.kept_segments[2].timeline_start == 5.0
    assert edl.kept_segments[2].timeline_end == 7.5

    assert edl.requires_human_approval is True
    assert edl.approval_status == "pending"

def test_remap_subtitles_for_edl(tmp_path):
    """Verifies that ASS subtitle events are remapped to the tightened timeline and cut words omitted."""
    tr = make_sample_transcript()
    c1 = EditDecision(
        cut_id="cut_01",
        cut_type="silence_trim",
        original_start=11.9,
        original_end=12.7,
        duration_cut=0.8,
        is_meaningful_content=False,
        rationale="Silence trim",
        approved=True,
    )
    edl = build_edit_decision_list(
        candidate_id="cand_test",
        start_time=10.0,
        end_time=14.0,
        proposed_cuts=[c1],
    )
    
    ass_file = tmp_path / "test_remap.ass"
    remap_subtitles_for_edl(tr, edl, str(ass_file))
    assert ass_file.exists()
    
    content = ass_file.read_text(encoding="utf-8")
    assert "Dialogue:" in content
    assert "wrong?" in content
    assert "works" in content

def test_approval_gate_blocks_unapproved_edl(tmp_path):
    """Verifies that render_tightened_short strictly blocks unapproved meaningful spoken cuts."""
    c_spoken = EditDecision(
        cut_id="cut_spoken",
        cut_type="repetition_removal",
        original_start=15.0,
        original_end=16.5,
        duration_cut=1.5,
        spoken_content="repeated words",
        is_meaningful_content=True,
        rationale="Remove repeated words",
        approved=False,
    )
    edl = build_edit_decision_list(
        candidate_id="cand_test",
        start_time=10.0,
        end_time=20.0,
        proposed_cuts=[c_spoken],
    )
    assert edl.requires_human_approval is True
    assert edl.approval_status == "pending"
    
    # Attempting to render an unapproved EDL must raise PermissionError
    with pytest.raises(PermissionError) as exc_info:
        render_tightened_short(
            source_video_path="nonexistent.mp4",
            edl=edl,
            output_mp4_path=str(tmp_path / "out.mp4"),
            test_mode=False,
        )
    assert "contains cuts of meaningful spoken content" in str(exc_info.value)

def test_edl_approval_mutation(tmp_path):
    """Verifies updating EDL approval status in candidates manifest."""
    manifest_file = tmp_path / "candidates.json"
    
    edl = EditDecisionList(
        edl_id="edl_candidate_02",
        candidate_id="candidate_02",
        original_start_time=208.68,
        original_end_time=232.80,
        original_duration=24.12,
        tightened_duration=22.50,
        duration_saved=1.62,
        cuts=[
            EditDecision(
                cut_id="cut_01",
                cut_type="repetition_removal",
                original_start=224.62,
                original_end=225.74,
                duration_cut=1.12,
                spoken_content="and that place of like,",
                is_meaningful_content=True,
                rationale="Remove repetition",
                approved=False,
            )
        ],
        requires_human_approval=True,
        approval_status="pending",
    )
    
    cand = ClipCandidate(
        candidate_id="candidate_02",
        start_time=208.68,
        end_time=232.80,
        duration=24.12,
        headline="Highlight: What if it all goes wrong?",
        hook_summary="What if it all goes wrong?",
        transcript_excerpt="What if it all goes wrong? What if it all works out?",
        transcript_segment_ids=[40, 41],
        observed_visual_evidence=ObservedVisualEvidence(),
        selection_rationale=SelectionRationale(
            hook_score=0.9, flow_coherence=0.9, visual_engagement=0.9, overall_score=0.85, justification="Test"
        ),
        confidence_score=0.85,
        approval_status="pending",
        edl=edl,
    )
    
    manifest = CandidatesManifest(
        source_video="test_vid",
        analysis_mode="interactive_antigravity",
        min_duration=20.0,
        max_duration=45.0,
        target_duration=30.0,
        total_candidates_found=1,
        candidates=[cand],
    )
    manifest_file.write_text(json.dumps(manifest.model_dump(), indent=2), encoding="utf-8")
    
    # Update EDL approval
    updated_edl = update_edl_approval(
        candidates_file=str(manifest_file),
        candidate_id="candidate_02",
        new_status="approved",
        reviewed_by="Human Lead",
    )
    assert updated_edl.approval_status == "approved"
    assert updated_edl.reviewed_by == "Human Lead"
    assert updated_edl.cuts[0].approved is True
