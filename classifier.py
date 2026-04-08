# classifier.py
import pandas as pd
import numpy as np
import joblib
import os

LABEL_ENCODER_PATH = "model/label_encoder.pkl"
CONFIDENCE_THRESHOLD = 55.0  # below this → Inconclusive

def load_stem_model(stem_key: str):
    model_path   = f"model/{stem_key}_model.pkl"
    scaler_path  = f"model/{stem_key}_scaler.pkl"
    feature_path = f"model/{stem_key}_features.pkl"

    # Fall back to vocals model if stem-specific doesn't exist
    if not os.path.exists(model_path):
        stem_key  = "vocals"
        model_path   = f"model/{stem_key}_model.pkl"
        scaler_path  = f"model/{stem_key}_scaler.pkl"
        feature_path = f"model/{stem_key}_features.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Models not found. Run train_stem_classifier.py first."
        )

    return (
        joblib.load(model_path),
        joblib.load(scaler_path),
        joblib.load(feature_path),
        joblib.load(LABEL_ENCODER_PATH)
    )


def predict_genre(features: dict, stem_key: str = "vocals") -> dict:
    if features is None:
        return None

    model, scaler, feature_cols, label_encoder = load_stem_model(stem_key)

    # ── Map our feature names to GTZAN column names ───────────
    FEATURE_NAME_MAP = {
        "mfcc_1":                  "mfcc1_mean",
        "mfcc_2":                  "mfcc2_mean",
        "mfcc_3":                  "mfcc3_mean",
        "mfcc_4":                  "mfcc4_mean",
        "mfcc_5":                  "mfcc5_mean",
        "mfcc_6":                  "mfcc6_mean",
        "mfcc_7":                  "mfcc7_mean",
        "mfcc_8":                  "mfcc8_mean",
        "mfcc_9":                  "mfcc9_mean",
        "mfcc_10":                 "mfcc10_mean",
        "mfcc_11":                 "mfcc11_mean",
        "mfcc_12":                 "mfcc12_mean",
        "mfcc_13":                 "mfcc13_mean",
        "spectral_centroid_mean":  "spectral_centroid_mean",
        "spectral_bandwidth_mean": "spectral_bandwidth_mean",
        "spectral_rolloff_mean":   "rolloff_mean",
        "zcr_mean":                "zero_crossing_rate_mean",
        "chroma_mean":             "chroma_stft_mean",
        "tempo":                   "tempo",
    }

    # ── Build row aligned to model's expected columns ─────────
    mapped = {}
    for our_key, gtzan_key in FEATURE_NAME_MAP.items():
        if our_key in features:
            mapped[gtzan_key] = features[our_key]

    row = {col: mapped.get(col, 0.0) for col in feature_cols}
    X   = pd.DataFrame([row])
    X_scaled = scaler.transform(X)

    # ── Get probabilities ─────────────────────────────────────
    proba  = model.predict_proba(X_scaled)[0]
    genres = label_encoder.classes_
    ranked = sorted(zip(genres, proba), key=lambda x: x[1], reverse=True)

    top_genre = ranked[0][0]
    top_conf  = round(ranked[0][1] * 100, 1)

    # ── Confidence threshold ──────────────────────────────────
    inconclusive = top_conf < CONFIDENCE_THRESHOLD

    return {
        "top_genre":      "Inconclusive" if inconclusive else top_genre,
        "confidence":     top_conf,
        "inconclusive":   inconclusive,
        "top_3": [
            {"genre": g, "confidence": round(c * 100, 1)}
            for g, c in ranked[:3]
        ],
        "all_genres": {
            g: round(c * 100, 1) for g, c in ranked
        },
        "note": (
            "Confidence too low for a reliable prediction — "
            "isolated stems differ significantly from the full-song "
            "training data." if inconclusive else None
        )
    }


def predict_all_stems(all_features: dict) -> dict:
    results = {}
    for stem_key, features in all_features.items():
        if features is not None:
            print(f"Predicting genre for: {stem_key}")
            results[stem_key] = predict_genre(features, stem_key)
        else:
            results[stem_key] = None
    return results

def get_confusion_matrix_data() -> dict:
    """
    Runs the model against the GTZAN test set and returns
    confusion matrix data for visualization in the app.
    """
    import json

    if not os.path.exists("features_30_sec.csv"):
        return None

    df = pd.read_csv("features_30_sec.csv")
    df = df.drop(columns=["filename", "length"], errors="ignore")
    X  = df.drop(columns=["label"])
    y  = df["label"]

    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    y_encoded     = label_encoder.transform(y)

    # Use vocals model as the general reference model
    model, scaler, feature_cols, _ = load_stem_model("vocals")

    row_data = {col: X[col].values if col in X.columns else 0.0
                for col in feature_cols}
    X_aligned = pd.DataFrame(row_data)[feature_cols]
    X_scaled  = scaler.transform(X_aligned)

    y_pred    = model.predict(X_scaled)
    classes   = label_encoder.classes_

    from sklearn.metrics import confusion_matrix as sk_cm
    cm = sk_cm(y_encoded, y_pred)

    return {
        "matrix":  cm.tolist(),
        "classes": list(classes),
        "accuracy": round((y_pred == y_encoded).mean() * 100, 1)
    }