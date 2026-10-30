"""AI Analysis and Highlight Selection Orchestrator for Matos Video Intelligence."""
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

from matos_video.config import (
    logger,
    TranscriptResult,
    TranscriptSegment,
    FrameInfo,
    ClipCandidate,
    ObservedVisualEvidence,
    SelectionRationale,
    CandidatesManifest,
)

HOOK_KEYWORDS = [
    "notice", "key", "secret", "critical", "why", "how", "what",
    "welcome", "today", "look", "important", "increase", "drop",
    "retention", "hook", "first", "tested", "proven", "revealed",
]

CONFUSING_ENUMERATION_OPENINGS = [
    "the first was", "the second was", "the third was",
    "number one", "number two", "number three",
    "secondly", "thirdly", "lastly", "first off",
    "second side of the fear", "first side of the fear",
]

SLOW_PREAMBLE_PATTERNS = [
    "a lot of people use the concept of",
    "you know,", "i think that", "so basically",
    "like i was saying", "two minutes to decide",
    "they're just no longer going to wait",
    "and that you are", "it was then that",
]

ABRUPT_ENDING_WORDS = [
    "and", "but", "so", "or", "if", "because", "then",
    "like", "to", "of", "with", "into", "for", "at",
    "by", "from", "on", "in", "which", "that", "the",
]

FILLER_WORDS = {
    "like", "um", "uh", "actually", "basically", "literally", "sort of", "kind of",
}

def score_hook(segment: TranscriptSegment) -> float:
    """Scores opening hook quality based on speech speed, confidence, and hook keywords,
    penalizing slow preambles, confusing enumeration, and unresolved context-dependent openings.
    NOTE: Heuristic scores are automated screening filters, NOT proof of editorial quality.
    """
    text_lower = segment.text.lower().strip()
    score = 0.50

    # 1. Check hook keywords in opening sentence
    matched_words = [kw for kw in HOOK_KEYWORDS if kw in text_lower]
    score += min(0.30, len(matched_words) * 0.10)

    # 2. Question marks or exclamations increase curiosity
    if "?" in segment.text or "!" in segment.text:
        score += 0.10

    # 3. High transcription probability in opening words
    if segment.words:
        first_few = segment.words[:3]
        avg_prob = sum(w.probability for w in first_few) / len(first_few)
        score += (avg_prob - 0.7) * 0.15

    # --- EDITORIAL PENALTIES ---
    # Penalty A: Confusing enumeration opening without context
    for pattern in CONFUSING_ENUMERATION_OPENINGS:
        if pattern in text_lower[:60]:
            score -= 0.35
            logger.debug(f"Applied confusing enumeration penalty (-0.35) for: '{pattern}'")
            break

    # Penalty B: Slow conversational preamble
    for preamble in SLOW_PREAMBLE_PATTERNS:
        if text_lower.startswith(preamble) or preamble in text_lower[:50]:
            score -= 0.25
            logger.debug(f"Applied slow preamble penalty (-0.25) for: '{preamble}'")
            break

    # Penalty C: Unresolved context-dependent opening pronouns
    words = text_lower.split()
    if words and words[0] in ["they're", "it", "that", "these", "those", "she", "he"]:
        score -= 0.15

    return round(max(0.1, min(1.0, score)), 3)

def score_coherence(segments: List[TranscriptSegment]) -> float:
    """Evaluates syntactic and conversational closure across a span of segments,
    heavily penalizing incomplete thoughts, trailing prepositions/conjunctions, and abrupt cutoffs.
    """
    if not segments:
        return 0.0

    score = 0.60
    first_seg = segments[0]
    last_seg = segments[-1]
    last_text = last_seg.text.strip()
    last_text_lower = last_text.lower()

    # Sentence starts cleanly with uppercase
    if first_seg.text and first_seg.text[0].isupper():
        score += 0.10

    # Sentence concludes with proper terminal punctuation
    if last_text.endswith((".", "!", "?")):
        score += 0.15
    elif last_text.endswith((",", ";", "-", ":")):
        score -= 0.30

    # Incomplete thought / abrupt ending check: ends on conjunction or preposition
    last_words = [w.strip(".,!?:;\"'() ") for w in last_text_lower.split() if w.strip(".,!?:;\"'() ")]
    if last_words and last_words[-1] in ABRUPT_ENDING_WORDS:
        score -= 0.45
        logger.debug(f"Applied abrupt ending penalty (-0.45) for trailing word: '{last_words[-1]}'")

    # Incomplete conditional clause check (e.g. 'if you don't have the discipline')
    if "if you " in last_text_lower and not any(p in last_text_lower for p in ["then", "you will", "you're"]):
        score -= 0.30

    # Average logprob check
    valid_logprobs = [s.avg_logprob for s in segments if s.avg_logprob is not None]
    if valid_logprobs:
        avg_lp = sum(valid_logprobs) / len(valid_logprobs)
        if avg_lp > -0.4:
            score += 0.05

    return round(max(0.1, min(1.0, score)), 3)

def compute_editorial_penalty(segments: List[TranscriptSegment]) -> float:
    """Calculates cumulative editorial penalties for filler density and verbal stumbles."""
    all_words = []
    for s in segments:
        for w in (s.words or []):
            clean = w.word.strip(".,!?:;\"'() ").lower()
            if clean:
                all_words.append(clean)

    if not all_words:
        return 0.0

    # Filler word ratio
    filler_count = sum(1 for w in all_words if w in FILLER_WORDS)
    density = filler_count / len(all_words)
    penalty = 0.0
    if density > 0.05:  # Over 5% filler words
        penalty += min(0.20, (density - 0.05) * 1.5)

    return round(penalty, 3)

def find_candidate_windows(
    transcript: TranscriptResult,
    frames: List[FrameInfo],
    min_duration: float = 20.0,
    max_duration: float = 45.0,
    target_duration: float = 30.0,
) -> List[ClipCandidate]:
    """Identifies highlight candidates by grouping transcript segments into coherent windows."""
    segments = transcript.segments
    candidates: List[ClipCandidate] = []

    # Edge case 1: No transcript segments (Silent / non-speech video)
    if not segments:
        logger.warning("No transcript segments found; falling back to visual activity windowing.")
        total_dur = transcript.duration_seconds
        clip_dur = min(max_duration, max(min_duration, total_dur))
        matching_frames = [f for f in frames if f.timestamp_seconds <= clip_dur]
        candidate = ClipCandidate(
            candidate_id="candidate_01",
            start_time=0.0,
            end_time=round(clip_dur, 2),
            duration=round(clip_dur, 2),
            headline="Visual Activity Highlight (No Dialogue Detected)",
            hook_summary="Non-verbal scene sequence",
            transcript_excerpt="[No dialogue detected in audio track]",
            transcript_segment_ids=[],
            observed_visual_evidence=ObservedVisualEvidence(
                keyframe_timestamps=[f.timestamp_seconds for f in matching_frames],
                keyframe_paths=[f.relative_path for f in matching_frames],
                visual_elements_detected=["Visual motion/scene changes detected"],
                unspoken_visual_facts=["Audio track contains no recognized speech."],
            ),
            selection_rationale=SelectionRationale(
                hook_score=0.4,
                flow_coherence=0.3,
                visual_engagement=0.6,
                overall_score=0.45,
                justification="Candidate formed from visual activity in the absence of dialogue.",
            ),
            confidence_score=0.5,
            uncertainty_notes="Audio track has no detected speech. Visual inspection required.",
            approval_status="pending",
        )
        return [candidate]

    # Edge case 2: Total video length is shorter than min_duration
    total_speech_dur = segments[-1].end - segments[0].start
    if total_speech_dur < min_duration:
        logger.info(f"Total speech duration ({total_speech_dur:.2f}s) is below min_duration ({min_duration}s). Producing full-segment candidate.")
        excerpt = " ".join(s.text for s in segments)
        matching_frames = [f for f in frames if segments[0].start <= f.timestamp_seconds <= segments[-1].end]
        candidate = ClipCandidate(
            candidate_id="candidate_01",
            start_time=segments[0].start,
            end_time=segments[-1].end,
            duration=round(total_speech_dur, 2),
            headline="Full Short Segment Overview",
            hook_summary=segments[0].text,
            transcript_excerpt=excerpt,
            transcript_segment_ids=[s.id for s in segments],
            observed_visual_evidence=ObservedVisualEvidence(
                keyframe_timestamps=[f.timestamp_seconds for f in matching_frames],
                keyframe_paths=[f.relative_path for f in matching_frames],
                visual_elements_detected=[],
                unspoken_visual_facts=[],
            ),
            selection_rationale=SelectionRationale(
                hook_score=score_hook(segments[0]),
                flow_coherence=score_coherence(segments),
                visual_engagement=0.7,
                overall_score=0.75,
                justification="Complete coherent passage; duration is below target threshold.",
            ),
            confidence_score=0.8,
            uncertainty_notes=f"Clip duration ({total_speech_dur:.2f}s) is shorter than target minimum ({min_duration}s).",
            approval_status="pending",
        )
        return [candidate]

    # Standard Windowing Search
    candidate_pools = []
    n = len(segments)

    for i in range(n):
        for j in range(i, n):
            start = segments[i].start
            end = segments[j].end
            dur = end - start

            if min_duration <= dur <= max_duration:
                hook = score_hook(segments[i])
                coherence = score_coherence(segments[i:j+1])
                editorial_pen = compute_editorial_penalty(segments[i:j+1])

                # Match frames
                sub_frames = [f for f in frames if start <= f.timestamp_seconds <= end]
                scene_cuts = sum(1 for f in sub_frames if f.is_scene_change)
                visual_eng = min(1.0, 0.4 + (scene_cuts * 0.2) + (len(sub_frames) * 0.03))

                # Penalty for deviating from target_duration
                dur_penalty = abs(dur - target_duration) / target_duration * 0.15
                
                # Combined automated score with realistic heuristic calibration cap (max 0.85).
                # Heuristic screening scores must not overestimate editorial quality.
                raw_score = (hook * 0.35) + (coherence * 0.40) + (visual_eng * 0.25) - dur_penalty - editorial_pen
                overall = round(max(0.1, min(0.85, raw_score)), 3)

                candidate_pools.append({
                    "start_idx": i,
                    "end_idx": j,
                    "start": start,
                    "end": end,
                    "duration": round(dur, 2),
                    "hook_score": hook,
                    "coherence": coherence,
                    "visual_eng": round(visual_eng, 3),
                    "overall": overall,
                    "frames": sub_frames,
                })

    # Sort pools by overall score descending
    candidate_pools.sort(key=lambda x: x["overall"], reverse=True)

    # Select top non-redundant candidates (max 3)
    selected_pools = []
    for pool in candidate_pools:
        # Check overlap with already selected
        overlap = False
        for s in selected_pools:
            inter_start = max(pool["start"], s["start"])
            inter_end = min(pool["end"], s["end"])
            if inter_end > inter_start:
                inter_dur = inter_end - inter_start
                # If overlap is > 65% of the shorter clip, consider redundant
                if inter_dur / min(pool["duration"], s["duration"]) > 0.65:
                    overlap = True
                    break
        if not overlap:
            selected_pools.append(pool)
        if len(selected_pools) >= 3:
            break

    # If no window strictly fell within [min, max], take the closest passage
    if not selected_pools and candidate_pools:
        selected_pools = [candidate_pools[0]]
    elif not selected_pools:
        # Fallback to entire transcript if nothing else matched
        selected_pools = [{
            "start_idx": 0,
            "end_idx": n - 1,
            "start": segments[0].start,
            "end": segments[-1].end,
            "duration": round(segments[-1].end - segments[0].start, 2),
            "hook_score": score_hook(segments[0]),
            "coherence": score_coherence(segments),
            "visual_eng": 0.7,
            "overall": 0.8,
            "frames": frames,
        }]

    # Convert pools to ClipCandidate models
    for idx, pool in enumerate(selected_pools, start=1):
        seg_subset = segments[pool["start_idx"] : pool["end_idx"] + 1]
        excerpt = " ".join(s.text for s in seg_subset)
        headline = seg_subset[0].text.split(".")[0].split(",")[0]
        if len(headline) > 60:
            headline = headline[:57] + "..."

        cand = ClipCandidate(
            candidate_id=f"candidate_{idx:02d}",
            start_time=pool["start"],
            end_time=pool["end"],
            duration=pool["duration"],
            headline=f"Highlight: {headline}",
            hook_summary=seg_subset[0].text[:90] + ("..." if len(seg_subset[0].text) > 90 else ""),
            transcript_excerpt=excerpt,
            transcript_segment_ids=[s.id for s in seg_subset],
            observed_visual_evidence=ObservedVisualEvidence(
                keyframe_timestamps=[f.timestamp_seconds for f in pool["frames"]],
                keyframe_paths=[f.relative_path for f in pool["frames"]],
                visual_elements_detected=[],
                unspoken_visual_facts=[],
            ),
            selection_rationale=SelectionRationale(
                hook_score=pool["hook_score"],
                flow_coherence=pool["coherence"],
                visual_engagement=pool["visual_eng"],
                overall_score=pool["overall"],
                justification=(
                    f"Clip spans {pool['duration']}s with a compelling opening hook ({pool['hook_score']}), "
                    f"grammatical boundary closure ({pool['coherence']}), and {len(pool['frames'])} timestamped frames."
                ),
            ),
            confidence_score=round(pool["overall"] * 0.95, 2),
            uncertainty_notes=None,
            approval_status="pending",
        )
        candidates.append(cand)

    logger.info(f"Identified {len(candidates)} highlight candidate(s).")
    return candidates

def attach_multimodal_evidence(
    candidate: ClipCandidate,
    visual_elements: List[str],
    unspoken_facts: List[str],
    notes: Optional[str] = None,
) -> ClipCandidate:
    """Attaches verified multimodal visual observations discovered by the AI agent."""
    candidate.observed_visual_evidence.visual_elements_detected = visual_elements
    candidate.observed_visual_evidence.unspoken_visual_facts = unspoken_facts
    if notes:
        candidate.uncertainty_notes = notes
    return candidate

def enrich_candidate_multimodal(
    candidates_path: str,
    candidate_id: str,
    visual_elements: List[str],
    unspoken_facts: List[str],
    notes: Optional[str] = None,
) -> CandidatesManifest:
    """Loads candidates.json, enriches specified candidate with multimodal evidence, and resaves manifest + report."""
    p = Path(candidates_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Candidates file not found: {candidates_path}")

    data = json.loads(p.read_text(encoding="utf-8"))
    manifest = CandidatesManifest.model_validate(data)

    target_candidate = None
    for c in manifest.candidates:
        if c.candidate_id == candidate_id:
            target_candidate = c
            break

    if not target_candidate:
        raise ValueError(f"Candidate '{candidate_id}' not found in manifest.")

    attach_multimodal_evidence(target_candidate, visual_elements, unspoken_facts, notes)

    # Re-export manifest and markdown report
    export_candidates_json(manifest, str(p))
    report_path = p.parent / "analysis_report.md"
    export_analysis_markdown_report(manifest, str(report_path))
    logger.info(f"Enriched candidate '{candidate_id}' with {len(unspoken_facts)} unspoken multimodal facts.")
    return manifest

def export_candidates_json(manifest: CandidatesManifest, output_path: str) -> str:
    """Exports structured CandidatesManifest to JSON."""
    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(manifest.model_dump(), indent=2), encoding="utf-8")
    logger.info(f"Exported candidates manifest to: {out_p}")
    return str(out_p)

def export_analysis_markdown_report(manifest: CandidatesManifest, output_path: str) -> str:
    """Generates an executive Markdown analysis report for operator review."""
    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Matos Video Intelligence: Clip Candidates & Visual Analysis Report",
        "",
        f"**Source Video**: `{manifest.source_video}`  ",
        f"**Analysis Mode**: `{manifest.analysis_mode}`  ",
        f"**Target Window**: {manifest.min_duration}s - {manifest.max_duration}s (Ideal: {manifest.target_duration}s)  ",
        f"**Candidates Identified**: {manifest.total_candidates_found}  ",
        "",
        "> [!IMPORTANT]",
        "> **Mandatory Human Approval**: No clips will be rendered into vertical Shorts without explicit approval status set to `approved`.",
        "",
        "> [!NOTE]",
        "> **Editorial Scoring Calibration**: Automated heuristic scores evaluate boundary punctuation, keyword density, and visual activity. They do not constitute proof of editorial excellence or storytelling impact. Human operator review is required to verify narrative flow, eliminate confusion, and approve internal cuts.",
        "",
        "---",
        "",
        "## Candidate Highlights Overview",
        "",
    ]

    for c in manifest.candidates:
        status_badge = "PENDING REVIEW"
        if c.approval_status == "approved":
            status_badge = "APPROVED"
        elif c.approval_status == "rejected":
            status_badge = "REJECTED"

        lines.extend([
            f"### [{c.candidate_id.upper()}] {c.headline}",
            "",
            f"- **Status**: `{status_badge}`",
            f"- **Time Range**: `{c.start_time:.2f}s` to `{c.end_time:.2f}s` (Duration: `{c.duration:.2f}s`)",
            f"- **Confidence Score**: `{c.confidence_score * 100:.1f}%`",
            f"- **Opening Hook**: *\"{c.hook_summary}\"*",
            "",
            "#### Selection Rationale & Scores",
            f"- **Hook Score**: `{c.selection_rationale.hook_score:.2f}` / 1.0",
            f"- **Flow & Closure**: `{c.selection_rationale.flow_coherence:.2f}` / 1.0",
            f"- **Visual Engagement**: `{c.selection_rationale.visual_engagement:.2f}` / 1.0",
            f"- **Overall Score**: `{c.selection_rationale.overall_score:.2f}` / 1.0",
            f"- **Justification**: {c.selection_rationale.justification}",
            "",
            "#### Spoken Transcript Excerpt",
            f"> \"{c.transcript_excerpt}\"",
            "",
            "#### Multimodal Visual Evidence (Observed in Extracted Frames)",
            "- **Elements Detected On Screen**:",
        ])

        for elem in c.observed_visual_evidence.visual_elements_detected:
            lines.append(f"  - {elem}")

        lines.append("- **Unspoken Visual Facts (Absent from Speech)**:")
        for fact in c.observed_visual_evidence.unspoken_visual_facts:
            lines.append(f"  - **{fact}**")

        if c.observed_visual_evidence.keyframe_paths:
            lines.append(f"- **Keyframe References**: {', '.join(f'`{k}`' for k in c.observed_visual_evidence.keyframe_paths[:4])}")

        if c.uncertainty_notes:
            lines.append(f"- **Operator Notes**: *{c.uncertainty_notes}*")

        lines.append("")
        lines.append("---")
        lines.append("")

    out_p.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Exported markdown analysis report to: {out_p}")
    return str(out_p)

def update_candidate_approval(
    candidates_file: str,
    candidate_id: str,
    new_status: str,
    reviewed_by: str = "human_operator",
) -> CandidatesManifest:
    """Updates candidate approval status ('approved' or 'rejected') and records review timestamp."""
    p = Path(candidates_file).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Candidates file not found: {candidates_file}")

    data = json.loads(p.read_text(encoding="utf-8"))
    manifest = CandidatesManifest.model_validate(data)

    import datetime
    now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()

    updated = False
    for c in manifest.candidates:
        if c.candidate_id == candidate_id:
            c.approval_status = new_status
            c.reviewed_by = reviewed_by
            c.review_timestamp = now_str
            updated = True
            break

    if not updated:
        raise ValueError(f"Candidate ID '{candidate_id}' not found in manifest.")

    p.write_text(json.dumps(manifest.model_dump(), indent=2), encoding="utf-8")
    logger.info(f"Updated candidate '{candidate_id}' status to '{new_status}' by '{reviewed_by}'")
    return manifest
