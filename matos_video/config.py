"""Configuration, system probes, and data models for Matos Video Intelligence."""
from pathlib import Path
import os
import shutil
import logging
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("matos_video")

# Standard paths
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = WORKSPACE_DIR / "outputs"
MODELS_DIR = WORKSPACE_DIR / "models"

# Detect FFmpeg and FFprobe binaries
def find_ffmpeg_binary(name: str = "ffmpeg") -> str:
    # 1. Try PATH
    exe_name = f"{name}.exe" if os.name == "nt" else name
    found = shutil.which(exe_name) or shutil.which(name)
    if found:
        return found

    # 2. Try known WinGet path on Windows
    if os.name == "nt":
        winget_pattern = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
        if winget_pattern.exists():
            matches = list(winget_pattern.glob(f"**/bin/{exe_name}"))
            if matches:
                return str(matches[0])
    
    raise FileNotFoundError(f"Could not locate {name} binary. Ensure FFmpeg is installed and added to PATH.")

FFMPEG_PATH = find_ffmpeg_binary("ffmpeg")
FFPROBE_PATH = find_ffmpeg_binary("ffprobe")

# Pydantic Data Models for structured output (n8n compatible)
class VideoStreamInfo(BaseModel):
    codec_name: str
    width: int
    height: int
    fps: float
    duration_seconds: float
    bitrate: Optional[int] = None

class AudioStreamInfo(BaseModel):
    codec_name: str
    channels: int
    sample_rate: int
    duration_seconds: float
    bitrate: Optional[int] = None

class VideoMetadata(BaseModel):
    file_path: str
    file_name: str
    file_size_bytes: int
    format_name: str
    duration_seconds: float
    video_stream: Optional[VideoStreamInfo] = None
    audio_stream: Optional[AudioStreamInfo] = None
    has_video: bool
    has_audio: bool

class WordTimestamp(BaseModel):
    word: str
    start: float
    end: float
    probability: float

class TranscriptSegment(BaseModel):
    id: int
    start: float
    end: float
    text: str
    words: List[WordTimestamp] = Field(default_factory=list)
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None

class TranscriptResult(BaseModel):
    language: str
    duration_seconds: float
    device_used: str
    compute_type: str
    model_name: str
    segments: List[TranscriptSegment]

class FrameInfo(BaseModel):
    frame_index: int
    timestamp_seconds: float
    file_path: str
    relative_path: str
    width: int
    height: int
    is_scene_change: bool = False
    scene_score: float = 0.0

class Phase2Manifest(BaseModel):
    video_metadata: VideoMetadata
    transcript: TranscriptResult
    audio_wav_path: str
    frames_directory: str
    total_frames_extracted: int
    frames: List[FrameInfo]

class ObservedVisualEvidence(BaseModel):
    keyframe_timestamps: List[float] = Field(default_factory=list)
    keyframe_paths: List[str] = Field(default_factory=list)
    visual_elements_detected: List[str] = Field(default_factory=list)
    unspoken_visual_facts: List[str] = Field(default_factory=list)

class SelectionRationale(BaseModel):
    hook_score: float
    flow_coherence: float
    visual_engagement: float
    overall_score: float
    justification: str

class EditDecision(BaseModel):
    cut_id: str
    cut_type: str  # "silence_trim", "filler_removal", "repetition_removal", "off_topic_removal", "pacing_adjustment"
    original_start: float
    original_end: float
    duration_cut: float
    spoken_content: Optional[str] = None
    is_meaningful_content: bool = False  # True if spoken words are removed; requires explicit human approval
    rationale: str
    approved: bool = False

class TightenedSegment(BaseModel):
    segment_index: int
    source_start: float
    source_end: float
    timeline_start: float
    timeline_end: float
    duration: float

class EditDecisionList(BaseModel):
    edl_id: str
    candidate_id: str
    original_start_time: float
    original_end_time: float
    original_duration: float
    tightened_duration: float
    duration_saved: float
    cuts: List[EditDecision] = Field(default_factory=list)
    kept_segments: List[TightenedSegment] = Field(default_factory=list)
    requires_human_approval: bool = False
    approval_status: str = "pending"  # "pending", "approved", "rejected"
    reviewed_by: Optional[str] = None
    review_timestamp: Optional[str] = None

class ClipCandidate(BaseModel):
    candidate_id: str
    start_time: float
    end_time: float
    duration: float
    headline: str
    hook_summary: str
    transcript_excerpt: str
    transcript_segment_ids: List[int] = Field(default_factory=list)
    observed_visual_evidence: ObservedVisualEvidence
    selection_rationale: SelectionRationale
    confidence_score: float
    uncertainty_notes: Optional[str] = None
    approval_status: str = "pending"  # "pending", "approved", "rejected"
    reviewed_by: Optional[str] = None
    review_timestamp: Optional[str] = None
    edl: Optional[EditDecisionList] = None  # Optional Smart Tightening Edit Decision List
    tightened_duration: Optional[float] = None

class CandidatesManifest(BaseModel):
    source_video: str
    analysis_mode: str
    min_duration: float
    max_duration: float
    target_duration: float
    total_candidates_found: int
    candidates: List[ClipCandidate]

class RenderedClip(BaseModel):
    clip_id: str
    candidate_id: str
    start_time: float
    end_time: float
    duration_seconds: float
    resolution: str
    crop_mode: str
    approval_status: str
    reviewed_by: Optional[str]
    review_timestamp: Optional[str]
    mp4_path: str
    file_size_bytes: int

class RenderManifest(BaseModel):
    source_video: str
    rendered_at: str
    crop_mode_default: str
    total_rendered: int
    rendered_clips: List[RenderedClip]
