"""
Emotion Mapping Engine for Dimensional ABSA (Valence & Arousal)
==============================================================
Maps continuous 2D (Valence, Arousal) coordinates in [0.000, 1.000] x [0.000, 1.000]
to fine-grained discrete emotion categories, Russell's Circumplex quadrants,
Geneva Emotion Wheel coordinates, intensity ratings, and HEX color themes.
"""

import math
from typing import Dict, Any, List, Tuple

# 24 Canonical Emotion Archetypes in (V_norm, A_norm) space where center is (0.5, 0.5)
# V in [0, 1] (0 = extremely unpleasant, 1 = extremely pleasant)
# A in [0, 1] (0 = extremely low arousal/passive, 1 = extremely high arousal/active)
EMOTION_TAXONOMY: List[Dict[str, Any]] = [
    # --- QUADRANT 1: High Valence, High Arousal (Positive & Active) ---
    {
        "name": "Ecstatic / Euphoric",
        "quadrant": "Q1 (Positive & High Energy)",
        "v": 0.95, "a": 0.90,
        "emoji": "🤩",
        "color": "#10B981", # Emerald green
        "description": "Intense delight, victory, peak euphoria or supreme joy."
    },
    {
        "name": "Joyful / Delighted",
        "quadrant": "Q1 (Positive & High Energy)",
        "v": 0.88, "a": 0.75,
        "emoji": "😄",
        "color": "#22C55E", # Bright green
        "description": "Strong happiness, cheerful appreciation, great satisfaction."
    },
    {
        "name": "Excited / Exhilarated",
        "quadrant": "Q1 (Positive & High Energy)",
        "v": 0.80, "a": 0.85,
        "emoji": "🔥",
        "color": "#F59E0B", # Amber
        "description": "High anticipation, thrilled energy, passionate excitement."
    },
    {
        "name": "Proud / Inspired",
        "quadrant": "Q1 (Positive & High Energy)",
        "v": 0.80, "a": 0.65,
        "emoji": "🌟",
        "color": "#84CC16", # Lime green
        "description": "Feelings of accomplishment, respect, or high motivation."
    },
    {
        "name": "Amused / Playful",
        "quadrant": "Q1 (Positive & High Energy)",
        "v": 0.75, "a": 0.60,
        "emoji": "😂",
        "color": "#34D399", # Mint
        "description": "Humor, fun, entertainment, lighthearted laughter."
    },
    {
        "name": "Astonished / Amazed",
        "quadrant": "Q1 (Positive & High Energy)",
        "v": 0.70, "a": 0.85,
        "emoji": "😲",
        "color": "#06B6D4", # Cyan
        "description": "Pleasant surprise, wonder, admiration at unexpected excellence."
    },

    # --- QUADRANT 2: Low Valence, High Arousal (Negative & Active) ---
    {
        "name": "Enraged / Furious",
        "quadrant": "Q2 (Negative & High Energy)",
        "v": 0.10, "a": 0.95,
        "emoji": "🤬",
        "color": "#DC2626", # Crimson Red
        "description": "Extreme anger, outrage, indignation, explosive resentment."
    },
    {
        "name": "Angry / Hostile",
        "quadrant": "Q2 (Negative & High Energy)",
        "v": 0.20, "a": 0.80,
        "emoji": "😡",
        "color": "#EF4444", # Bright Red
        "description": "Strong displeasure, sharp condemnation, bitterness."
    },
    {
        "name": "Frustrated / Annoyed",
        "quadrant": "Q2 (Negative & High Energy)",
        "v": 0.30, "a": 0.70,
        "emoji": "😤",
        "color": "#F97316", # Deep Orange
        "description": "Irritation, thwarted expectations, grievance over failure."
    },
    {
        "name": "Disgusted / Repulsed",
        "quadrant": "Q2 (Negative & High Energy)",
        "v": 0.15, "a": 0.65,
        "emoji": "🤮",
        "color": "#9333EA", # Deep Purple
        "description": "Strong aversion, distaste, moral or sensory rejection."
    },
    {
        "name": "Anxious / Panicked",
        "quadrant": "Q2 (Negative & High Energy)",
        "v": 0.25, "a": 0.85,
        "emoji": "😰",
        "color": "#E11D48", # Rose
        "description": "High apprehension, dread, fear of adverse outcomes."
    },
    {
        "name": "Shocked / Alarmed",
        "quadrant": "Q2 (Negative & High Energy)",
        "v": 0.30, "a": 0.90,
        "emoji": "😱",
        "color": "#EA580C", # Flame Orange
        "description": "Unpleasant astonishment, distress, sudden scandal or crisis."
    },

    # --- QUADRANT 3: Low Valence, Low Arousal (Negative & Passive) ---
    {
        "name": "Depressed / Despairing",
        "quadrant": "Q3 (Negative & Low Energy)",
        "v": 0.10, "a": 0.15,
        "emoji": "😞",
        "color": "#475569", # Slate Grey
        "description": "Profound sadness, helplessness, deep dejection."
    },
    {
        "name": "Sad / Heartbroken",
        "quadrant": "Q3 (Negative & Low Energy)",
        "v": 0.20, "a": 0.30,
        "emoji": "😢",
        "color": "#64748B", # Steel Blue
        "description": "Sorrow, emotional pain, loss, poignant melancholy."
    },
    {
        "name": "Disappointed / Let Down",
        "quadrant": "Q3 (Negative & Low Energy)",
        "v": 0.32, "a": 0.40,
        "emoji": "😔",
        "color": "#D97706", # Brownish Amber
        "description": "Unmet expectations, sadness over suboptimal performance."
    },
    {
        "name": "Bored / Apathetic",
        "quadrant": "Q3 (Negative & Low Energy)",
        "v": 0.38, "a": 0.18,
        "emoji": "🥱",
        "color": "#94A3B8", # Light Slate
        "description": "Lack of interest, dullness, monotone indifference."
    },
    {
        "name": "Tired / Exhausted",
        "quadrant": "Q3 (Negative & Low Energy)",
        "v": 0.35, "a": 0.25,
        "emoji": "😩",
        "color": "#78716C", # Stone
        "description": "Weariness, mental fatigue, drained patience."
    },

    # --- QUADRANT 4: High Valence, Low Arousal (Positive & Passive) ---
    {
        "name": "Serene / Blissful",
        "quadrant": "Q4 (Positive & Low Energy)",
        "v": 0.90, "a": 0.20,
        "emoji": "😇",
        "color": "#3B82F6", # Sky Blue
        "description": "Deep peace, untroubled tranquility, pure bliss."
    },
    {
        "name": "Calm / Peaceful",
        "quadrant": "Q4 (Positive & Low Energy)",
        "v": 0.80, "a": 0.25,
        "emoji": "😌",
        "color": "#0EA5E9", # Cerulean Blue
        "description": "Quiet satisfaction, absence of stress, harmonious ease."
    },
    {
        "name": "Content / Satisfied",
        "quadrant": "Q4 (Positive & Low Energy)",
        "v": 0.75, "a": 0.40,
        "emoji": "😊",
        "color": "#14B8A6", # Teal
        "description": "Pleasant comfort, fulfilled expectations, quiet happiness."
    },
    {
        "name": "Relieved / Reassured",
        "quadrant": "Q4 (Positive & Low Energy)",
        "v": 0.72, "a": 0.35,
        "emoji": "😮‍💨",
        "color": "#065F46", # Emerald Dark
        "description": "Removal of tension or anxiety, feeling safe and settled."
    },
    {
        "name": "Grateful / Appreciative",
        "quadrant": "Q4 (Positive & Low Energy)",
        "v": 0.82, "a": 0.45,
        "emoji": "🙏",
        "color": "#10B981", # Emerald
        "description": "Thankfulness, warm recognition of good deed or service."
    },

    # --- CENTER / NEUTRAL REGION ---
    {
        "name": "Neutral / Balanced",
        "quadrant": "Neutral Zone",
        "v": 0.50, "a": 0.50,
        "emoji": "😐",
        "color": "#6B7280", # Gray
        "description": "Objective, factual, absence of strong emotion."
    }
]


def map_valence_arousal_to_emotion(v: float, a: float) -> Dict[str, Any]:
    """
    Finds the best matching fine-grained emotion archetype using weighted Euclidean distance
    and circumplex angle on the continuous (Valence, Arousal) plane.
    """
    v = max(0.0, min(1.0, float(v)))
    a = max(0.0, min(1.0, float(a)))

    # Compute intensity from neutral center (0.5, 0.5)
    dv = v - 0.5
    da = a - 0.5
    euclidean_radius = math.sqrt(dv**2 + da**2)
    # Max possible distance from center to a corner is sqrt(0.5^2 + 0.5^2) = 0.7071
    intensity_pct = min(100.0, (euclidean_radius / 0.7071) * 100.0)

    # If within ultra-neutral radius
    if euclidean_radius < 0.08:
        neutral_entry = [e for e in EMOTION_TAXONOMY if e["name"] == "Neutral / Balanced"][0]
        return {
            "emotion_name": neutral_entry["name"],
            "quadrant": neutral_entry["quadrant"],
            "emoji": neutral_entry["emoji"],
            "color": neutral_entry["color"],
            "description": neutral_entry["description"],
            "valence": round(v, 4),
            "arousal": round(a, 4),
            "intensity_percent": round(intensity_pct, 1),
            "circumplex_angle_deg": 0.0,
            "top_3_candidates": [neutral_entry["name"]]
        }

    # Circumplex angle in degrees [0, 360)
    angle_rad = math.atan2(da, dv)
    angle_deg = (math.degrees(angle_rad) + 360.0) % 360.0

    # Calculate distance to all taxonomy entries
    scored = []
    for item in EMOTION_TAXONOMY:
        if item["name"] == "Neutral / Balanced":
            continue
        dist = math.sqrt((item["v"] - v)**2 + (item["a"] - a)**2)
        scored.append((dist, item))

    scored.sort(key=lambda x: x[0])
    best_item = scored[0][1]
    top_3 = [s[1]["name"] for s in scored[:3]]

    # Determine general quadrant
    if v >= 0.50 and a >= 0.50:
        quad_str = "Q1: Positive & High Arousal (Active Positive)"
    elif v < 0.50 and a >= 0.50:
        quad_str = "Q2: Negative & High Arousal (Active Negative)"
    elif v < 0.50 and a < 0.50:
        quad_str = "Q3: Negative & Low Arousal (Passive Negative)"
    else:
        quad_str = "Q4: Positive & Low Arousal (Passive Positive)"

    return {
        "emotion_name": best_item["name"],
        "quadrant": quad_str,
        "emoji": best_item["emoji"],
        "color": best_item["color"],
        "description": best_item["description"],
        "valence": round(v, 4),
        "arousal": round(a, 4),
        "intensity_percent": round(intensity_pct, 1),
        "circumplex_angle_deg": round(angle_deg, 1),
        "top_3_candidates": top_3
    }
