# Matos Video Intelligence

> **Local-first AI video analysis and Shorts editing toolkit with multimodal insights, smart clipping, FFmpeg rendering, synchronized captions, and human approval.**

Matos Video Intelligence is a privacy-first, local AI automation system engineered for Matos Systems. It analyzes long-form landscape video (podcasts, webinars, product demos, interviews), discovers high-retention highlights through combined transcript heuristics and multimodal visual inspection, tightens pacing with automatic dead-air trimming and jump cuts, and renders polished 9:16 vertical Shorts with animated word-level highlighting.

---

## Key Features

- **100% Local-First Engine**: Zero mandatory cloud dependencies. Uses local FFmpeg/FFprobe binaries, local Faster-Whisper (`int8` CPU / CUDA), OpenCV, and PyAV.
- **Genuine Multimodal Visual Inspection**: The active AI pair programmer inspects extracted keyframes to detect speaker gestures, set context, eye contact, and emotional resonance absent from the spoken transcript.
- **Smart Tightening & Edit Decision Lists (EDL)**: Automatically detects pauses $> 0.5$s and trims dead air down to natural $0.25$s breath pauses. Identifies verbal stumbles and repetitive phrases, assembling multi-segment jump cuts via sample-accurate FFmpeg `concat` filtergraphs.
- **Calibrated Editorial Scoring**: Penalizes confusing list openings (*"The first was..."*), conversational preambles (*"A lot of people use the concept of..."*), abrupt sentence cutoffs, and excessive filler words. Caps automated heuristic scores at $0.85$ max so mechanical metrics are never confused with human editorial excellence.
- **Mandatory Human Approval Gate**: Unapproved candidates and edit decisions strictly block video rendering. Separates test harness approvals from human approvals.
- **Dynamic 9:16 Vertical Framing Modes**:
  - `pad_blur`: Non-destructive centered 16:9 card with ambient blurred background.
  - `pad_elevated`: Upper-card layout (Y=420) leaving a dedicated lower third for subtitles.
  - `zoom_blur`: 1.25x foreground zoom over ambient blur for higher speaker presence.
  - `center`: Full-bleed 1080x1920 portrait crop (ideal for centered talking heads).
  - `smart_crop`: Full-bleed portrait crop biased toward OpenCV edge-energy centroid.
- **Synchronized ASS Word-Highlight Subtitles**: Custom Advanced SubStation Alpha (`.ass`) subtitle generation with active yellow word-by-word highlight animation, automatically remapped across internal jump cuts.
- **Global Antigravity Skill**: Installed globally at `~/.gemini/config/skills/matos-video-intelligence` for invocation from any project or workspace.

---

## Architecture Overview

```
                      +---------------------------------------+
                      |       Long-Form Video Input           |
                      |   (Local MP4/MOV or Authorized URL)   |
                      +-------------------+-------------------+
                                          |
                      +-------------------+-------------------+
                      |             Intake Engine             |
                      |   FFprobe Stream Probe & yt-dlp Intake|
                      +-------------------+-------------------+
                                          |
                   +----------------------+----------------------+
                   |                                             |
                   v                                             v
     +---------------------------+                 +---------------------------+
     |   Audio & Transcription   |                 | Intelligent Frame Engine  |
     | Faster-Whisper CPU (int8) |                 |  Scene Transition & 2.0s  |
     |   Word-Level Timestamps   |                 |   Interval Keyframe Grabs |
     +-------------+-------------+                 +-------------+-------------+
                   |                                             |
                   +----------------------+----------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |         AI Analysis Orchestrator      |
                      | - Calibrated Hook & Coherence Scoring |
                      | - Penalties: Preamble, Lists, Fillers |
                      | - Multimodal Visual Understanding     |
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |      Optional Smart Tightening        |
                      | - Silence & Dead Air Trimming (0.25s) |
                      | - Repetition & Verbal Stumble Removal |
                      | - Edit Decision List (EDL) Generation |
                      | - Non-Contiguous Subtitle Remapping   |
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |      MANDATORY HUMAN APPROVAL GATE    |
                      |  Operators review and approve in CLI  |
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |     FFmpeg 9:16 Rendering Engine      |
                      | - Multi-segment trim/atrim + concat   |
                      | - pad_elevated / center / pad_blur    |
                      | - Word-highlight ASS subtitle burn-in |
                      | - Faststart MOOV atom validation      |
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |    1080x1920 MP4 Social Ready Short   |
                      +---------------------------------------+
```

---

## Installation & Setup

### Prerequisites
1. **Windows 10/11** (or Linux/macOS)
2. **Python 3.12+**
3. **FFmpeg & FFprobe**: Installed and available on system PATH (or installed via `winget install Gyan.FFmpeg`).

### Local Project Setup
```powershell
# Clone the repository
git clone https://github.com/matos-systems/matos-video-intelligence.git
cd matos-video-intelligence

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install package dependencies
pip install -e .
```

---

## CLI Usage Guide

The toolkit provides a modular CLI interface ready for terminal use or automated n8n sub-process execution:

### 1. Probe Video Metadata
```powershell
python -m matos_video.cli probe --input "path/to/video.mp4" --json
```

### 2. Download Video via yt-dlp
```powershell
python -m matos_video.cli download --url "https://youtube.com/watch?v=..." --output-dir "downloads"
```

### 3. Run Complete Intake & Extraction Pipeline
Extracts 16kHz audio, runs Faster-Whisper, and extracts scene-change keyframes:
```powershell
python -m matos_video.cli pipeline --input "path/to/video.mp4" --output-dir "outputs/my_video" --model base
```

### 4. Highlight Analysis & Candidate Selection
```powershell
python -m matos_video.cli analyze --session-dir "outputs/my_video" --min-duration 20 --max-duration 45 --target-duration 30
```
Generates `outputs/my_video/candidates.json` and `outputs/my_video/analysis_report.md`.

### 5. Generate Smart Tightening EDL
Detects dead-air pauses and stumbling phrase repetitions:
```powershell
python -m matos_video.cli tighten --session-dir "outputs/my_video" --candidate-id candidate_02 --min-pause 0.5 --retain-pause 0.25
```
Generates `outputs/my_video/edl_candidate_02.md` and attaches EDL to `candidates.json`.

### 6. Human Approval
Explicitly approve candidate selection or EDL:
```powershell
# Approve candidate
python -m matos_video.cli approve --candidates "outputs/my_video/candidates.json" --id candidate_02 --status approved --by human_operator

# Approve Smart Tightening EDL
python -m matos_video.cli approve --candidates "outputs/my_video/candidates.json" --id candidate_02 --status approved --by human_operator --edl
```

### 7. Render Vertical Short
```powershell
# Render original continuous cut
python -m matos_video.cli render --session-dir "outputs/my_video" --candidate-id candidate_02 --crop-mode center

# Render Smart-Tightened multi-segment cut
python -m matos_video.cli render --session-dir "outputs/my_video" --candidate-id candidate_02 --crop-mode center --tightened
```

---

## Global Antigravity Skill Invocation

Matos Video Intelligence is installed globally as an Antigravity skill at:
`C:\Users\Admin\.gemini\config\skills\matos-video-intelligence`

Invoke the tool from any workspace or directory without copying repository files:

```powershell
# Via universal Python runner
python "C:\Users\Admin\.gemini\config\skills\matos-video-intelligence\scripts\matos_runner.py" probe --input "C:\path\to\video.mp4"

# Via Windows batch script
& "C:\Users\Admin\.gemini\config\skills\matos-video-intelligence\scripts\matos.bat" pipeline --input "C:\path\to\video.mp4" --output-dir "outputs\session"
```

> **Cross-Machine Note**: The global runner references the local project installation at `c:\Users\Admin\OneDrive\Documents\Coding\matos-video-intelligence` by default. When configuring on another workstation, set the `MATOS_VIDEO_DIR` environment variable to point to the cloned repository path:
> ```powershell
> [System.Environment]::SetEnvironmentVariable('MATOS_VIDEO_DIR', 'C:\path\to\matos-video-intelligence', 'User')
> ```

---

## Testing & Quality Assurance

The codebase includes comprehensive unit and integration tests covering transcription, frame extraction, cropper filtergraphs, subtitle remapping, silence trimming, EDL calculation, and approval gates.

```powershell
.\.venv\Scripts\pytest.exe -v
```

**Test Verification Status**:
- **31 passing tests** across 7 test suites (`test_analyzer.py`, `test_tightener.py`, `test_editor.py`, `test_renderer.py`, `test_frames.py`, `test_intake.py`, `test_transcribe.py`).
- Runtime: ~8.9s on Windows.

---

## Known Limitations

1. **Audio Re-encoding**: AAC audio re-encoded at 192kbps/48kHz decompresses to PCM and recompresses, which is lossy and does not preserve original audio bit-for-bit.
2. **Jump Cut Visual Transitions**: Removing 1.0s+ of spoken dialogue in single-camera footage creates a head jump cut. Natural conversational rhythm is preserved, but cross-dissolve/zoom punch-ins are not yet applied.
3. **Smart Crop Heuristic**: `smart_crop` relies on OpenCV Canny edge-energy centroids to bias framing ($X=0.25$ to $0.75$). It does not use neural face detection or dynamic keyframe tracking.

---

## Roadmap & Deferred Scope

The following capabilities are explicitly deferred to **December 2026 / January 2027**:

- **Automated Social Publishing**: Direct API publishing to YouTube Shorts, Instagram Reels, and TikTok via OAuth integrations.
- **Advanced Neural AI Face Tracking**: Deep learning facial landmark tracking (MediaPipe / InsightFace) with smooth motion panning.
- **Headless n8n Content Factory Orchestration**: Full webhook-triggered pipeline integration with automated queue processing and remote storage synchronization.

---

## License

Internal automation toolkit developed for Matos Systems. Strictly for authorized and owned content processing. All rights reserved.
