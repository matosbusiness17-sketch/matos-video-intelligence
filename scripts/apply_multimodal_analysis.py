"""Applies interactive Antigravity multimodal visual analysis findings to candidates manifest."""
from matos_video.analyzer import enrich_candidate_multimodal

candidates_file = "outputs/matos_sample_talk/candidates.json"

# Enrich Candidate 1
enrich_candidate_multimodal(
    candidates_path=candidates_file,
    candidate_id="candidate_01",
    visual_elements=[
        "Slide 2 (t=9.39s-14.0s): Dark card layout titled 'AUDIENCE ENGAGEMENT ANALYSIS' with orange metric badge 'METRIC ID: ENG-942'.",
        "Three visual metric cards displaying 'CONVERSION LIFT: +45.2%', 'AVERAGE RETENTION: 82.4%', and 'DROP-OFF REDUCTION: -64.0%'.",
        "Slide 3 (t=14.0s-22.0s): Red alert banner 'CRITICAL ATTENTION ZONE' with cards defining 'HOOK WINDOW (0.0s - 3.0s)', 'OPTIMAL CLIP LENGTH (30 - 45 sec)', and 'VERTICAL RE-FRAME 9:16 (1080x1920)'.",
        "Slide 4 (t=22.0s-30.88s): Architectural workflow diagram titled 'AUTONOMOUS CONTENT FACTORY PIPELINE' featuring 4 linked technical nodes with blue badge 'NODE WORKFLOW READY'."
    ],
    unspoken_facts=[
        "Identifier tag 'METRIC ID: ENG-942' is visibly displayed on screen but omitted from spoken dialogue.",
        "Engagement chart explicitly displays 'AVERAGE RETENTION: 82.4%' and 'DROP-OFF REDUCTION: -64.0%', whereas speech only references the 45% lift.",
        "Retention optimization slide specifies exact target format 'VERTICAL RE-FRAME: 9:16 (1080x1920)' and 'OPTIMAL CLIP LENGTH: 30 - 45 sec', which are unmentioned in audio.",
        "Architecture diagram details four discrete execution phases ('1. AUDIO SPEECH', '2. VISUAL FRAMES', '3. MULTIMODAL AI', '4. EXPORT BUNDLE') under 'Self-Hosted n8n Workflow Integration Schema'."
    ],
    notes="High visual diversity spanning 3 distinct presentation slides and 11 keyframes. Clear spoken hook in first 3 seconds."
)

# Enrich Candidate 2
enrich_candidate_multimodal(
    candidates_path=candidates_file,
    candidate_id="candidate_02",
    visual_elements=[
        "Slide 1 (t=0.0s-6.5s): Corporate title card 'MATOS VIDEO INTELLIGENCE' with green badge 'STATUS: PRODUCTION READY'.",
        "Hardware compute specification card 'COMPUTE STACK: RTX 5070 / i7' with subtext 'Blackwell + 20-Core Ultra'.",
        "Target architecture card 'LOCAL-FIRST: Zero-Cloud Dependence'.",
        "Slide 2 (t=6.5s-14.0s): Engagement metrics dashboard with orange badge 'METRIC ID: ENG-942' and conversion cards."
    ],
    unspoken_facts=[
        "Hardware stack 'RTX 5070 / i7 (Blackwell + 20-Core Ultra)' is explicitly documented on screen but never spoken in audio.",
        "Production status badge 'STATUS: PRODUCTION READY' is visible on Slide 1 without audio mention.",
        "System constraint 'Zero-Cloud Dependence' is highlighted under 'TARGET ARCHITECTURE' on Slide 1."
    ],
    notes="Introductory segment with strong brand presence. Visual transitions at t=6.5s and t=14.0s."
)

print("Multimodal evidence successfully applied to candidates.json and analysis_report.md")
