"""Headless Gemini Vision API connector for automated n8n workflows.
Separates interactive Antigravity visual analysis from external API execution.
"""
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from matos_video.config import logger

def is_headless_api_available() -> bool:
    """Returns True if GEMINI_API_KEY is configured in the environment."""
    return bool(os.environ.get("GEMINI_API_KEY"))

def analyze_frames_headless_api(
    frame_paths: List[str],
    transcript_excerpt: str,
    model_name: str = "gemini-2.5-flash",
) -> Dict[str, Any]:
    """Invokes Gemini API for headless multimodal frame evaluation if API key is present.
    Returns structured visual findings, or clear fallback status if key is missing."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {
            "available": False,
            "mode": "headless_gemini_api",
            "reason": (
                "GEMINI_API_KEY is not configured. Headless automated visual API analysis is disabled. "
                "Interactive Antigravity visual analysis or local heuristics will be utilized instead."
            ),
            "visual_elements_detected": [],
            "unspoken_visual_facts": [],
        }

    try:
        # Pluggable import: only loaded when API key is provided
        from google import genai
        from google.genai import types
        from PIL import Image

        client = genai.Client(api_key=api_key)
        images = []
        for p in frame_paths[:4]:  # Limit to top 4 keyframes to conserve free-tier tokens
            resolved_p = Path(p).resolve()
            if resolved_p.exists():
                images.append(Image.open(str(resolved_p)))

        prompt = (
            "You are an AI video intelligence analyzer. Inspect these video frames and the spoken transcript.\n"
            f"Spoken Transcript: \"{transcript_excerpt}\"\n\n"
            "Identify:\n"
            "1. Visual elements visible on screen (titles, badges, metric cards, diagrams, UI).\n"
            "2. Unspoken visual facts (factual information, numbers, IDs, or metrics visible on screen that are NOT mentioned in the spoken words).\n"
            "Return valid JSON with keys 'visual_elements_detected' (list of strings) and 'unspoken_visual_facts' (list of strings)."
        )

        response = client.models.generate_content(
            model=model_name,
            contents=[*images, prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )

        parsed = json.loads(response.text)
        return {
            "available": True,
            "mode": "headless_gemini_api",
            "visual_elements_detected": parsed.get("visual_elements_detected", []),
            "unspoken_visual_facts": parsed.get("unspoken_visual_facts", []),
        }
    except Exception as e:
        logger.warning(f"Headless Gemini API execution failed: {e}. Falling back gracefully.")
        return {
            "available": False,
            "mode": "headless_gemini_api",
            "reason": f"API execution failed: {e}",
            "visual_elements_detected": [],
            "unspoken_visual_facts": [],
        }
