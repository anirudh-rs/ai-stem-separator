# similarity.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

SIMILARITY_THRESHOLD = 75.0

STEM_WEIGHTS = {
    "vocals": 0.35,
    "drums":  0.15,
    "bass":   0.15,
    "guitar": 0.10,
    "piano":  0.20,
    "other":  0.05,
}

# ── Mean + variance features for better discrimination ────────
FEATURE_KEYS = [
    # Timbral fingerprint — mean AND std
    "mfcc_1",  "mfcc_2",  "mfcc_3",  "mfcc_4",  "mfcc_5",
    "mfcc_6",  "mfcc_7",  "mfcc_8",  "mfcc_9",  "mfcc_10",
    "mfcc_11", "mfcc_12", "mfcc_13",
    # Energy dynamics
    "energy_mean", "energy_max", "energy_std",
    # Spectral shape
    "spectral_centroid_mean", "spectral_rolloff_mean",
    "spectral_bandwidth_mean",
    # Harmonic content
    "chroma_mean", "chroma_std",
    # Rhythm
    "tempo", "beat_count",
    # Percussiveness
    "zcr_mean",
    # Silence pattern
    "silence_ratio",
]


def features_to_vector(features: dict) -> np.ndarray:
    return np.array([
        float(features.get(k, 0.0) or 0.0)
        for k in FEATURE_KEYS
    ]).reshape(1, -1)


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """Min-max normalize to remove scale bias between features."""
    min_val = vec.min()
    max_val = vec.max()
    if max_val - min_val == 0:
        return vec
    return (vec - min_val) / (max_val - min_val)


def compare_stems(features_a: dict, features_b: dict) -> dict:
    results        = {}
    weighted_total = 0.0
    weight_sum     = 0.0

    for stem in ["vocals", "drums", "bass", "guitar", "piano", "other"]:
        f_a = features_a.get(stem)
        f_b = features_b.get(stem)

        if f_a is None or f_b is None:
            results[stem] = {
                "score":   None,
                "flagged": False,
                "note":    "Not detected in one or both songs"
            }
            continue

        vec_a = normalize_vector(features_to_vector(f_a))
        vec_b = normalize_vector(features_to_vector(f_b))

        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a == 0 or norm_b == 0:
            results[stem] = {
                "score":   0.0,
                "flagged": False,
                "note":    "Silent stem"
            }
            continue

        # ── Cosine similarity ─────────────────────────────────
        cos_score = float(cosine_similarity(vec_a, vec_b)[0][0])

        # ── Euclidean distance penalty ────────────────────────
        # Penalizes vectors that are similarly shaped but
        # differ significantly in magnitude — catches cases
        # where two quiet ballads look identical to cosine
        euclidean_dist  = np.linalg.norm(vec_a - vec_b)
        max_dist        = np.sqrt(len(FEATURE_KEYS))
        distance_penalty = euclidean_dist / max_dist

        # Blend cosine similarity with distance penalty
        # 70% cosine shape + 30% euclidean distance
        blended = (0.7 * cos_score) + (0.3 * (1 - distance_penalty))
        score   = round(max(0.0, min(blended, 1.0)) * 100, 1)
        flagged = score >= SIMILARITY_THRESHOLD

        weight          = STEM_WEIGHTS.get(stem, 0.05)
        weighted_total += score * weight
        weight_sum     += weight

        results[stem] = {
            "score":   score,
            "flagged": flagged,
            "note": (
                "⚠️ High similarity — may indicate shared musical DNA"
                if flagged else "Within normal range"
            )
        }

    # ── Overall weighted score ────────────────────────────────
    overall         = round(weighted_total / weight_sum, 1) if weight_sum > 0 else 0.0
    overall_flagged = overall >= SIMILARITY_THRESHOLD

    return {
        "stems":           results,
        "overall_score":   overall,
        "overall_flagged": overall_flagged,
        "verdict": (
            "⚠️ High overall similarity detected"
            if overall_flagged
            else "✅ Songs appear sufficiently distinct"
        )
    }