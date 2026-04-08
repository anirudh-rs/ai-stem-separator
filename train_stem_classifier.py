# train_stem_classifier.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import xgboost as xgb
import joblib
import os
import json

print("Loading GTZAN dataset...")
df = pd.read_csv("features_30_sec.csv")
df = df.drop(columns=["filename", "length"], errors="ignore")

X = df.drop(columns=["label"])
y = df["label"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"Genres: {list(le.classes_)}")
print(f"Samples: {len(X)}")

os.makedirs("model", exist_ok=True)

# ── Stem feature profiles ─────────────────────────────────────
# Each stem type uses the features most relevant to it
STEM_FEATURE_GROUPS = {
    "vocals": [
        "mfcc1_mean", "mfcc2_mean", "mfcc3_mean", "mfcc4_mean",
        "mfcc5_mean", "mfcc6_mean", "mfcc7_mean", "mfcc8_mean",
        "mfcc9_mean", "mfcc10_mean", "mfcc11_mean", "mfcc12_mean", "mfcc13_mean",
        "spectral_centroid_mean", "spectral_bandwidth_mean",
        "rolloff_mean", "zero_crossing_rate_mean", "chroma_stft_mean"
    ],
    "drums": [
        "tempo", "beats",
        "zero_crossing_rate_mean", "zero_crossing_rate_var",
        "rmse_mean", "rmse_var",
        "spectral_centroid_mean", "spectral_bandwidth_mean",
        "mfcc1_mean", "mfcc2_mean", "mfcc3_mean"
    ],
    "bass": [
        "spectral_centroid_mean", "spectral_bandwidth_mean",
        "rolloff_mean", "rmse_mean",
        "mfcc1_mean", "mfcc2_mean", "mfcc3_mean", "mfcc4_mean",
        "chroma_stft_mean", "harmony_mean"
    ],
    "guitar": [
        "chroma_stft_mean", "chroma_stft_var",
        "spectral_centroid_mean", "spectral_bandwidth_mean",
        "mfcc1_mean", "mfcc2_mean", "mfcc3_mean",
        "mfcc4_mean", "mfcc5_mean",
        "zero_crossing_rate_mean", "tempo"
    ],
    "piano": [
        "chroma_stft_mean", "chroma_stft_var",
        "harmony_mean", "harmony_var",
        "spectral_centroid_mean", "spectral_bandwidth_mean",
        "mfcc1_mean", "mfcc2_mean", "mfcc3_mean",
        "mfcc4_mean", "mfcc5_mean", "mfcc6_mean"
    ],
    "other": [
        "mfcc1_mean", "mfcc2_mean", "mfcc3_mean", "mfcc4_mean",
        "mfcc5_mean", "mfcc6_mean", "mfcc7_mean", "mfcc8_mean",
        "spectral_centroid_mean", "spectral_bandwidth_mean",
        "rolloff_mean", "chroma_stft_mean", "tempo"
    ]
}

# ── Map our feature names to GTZAN column names ───────────────
FEATURE_MAP = {
    "mfcc1_mean":                  "mfcc1_mean",
    "mfcc2_mean":                  "mfcc2_mean",
    "mfcc3_mean":                  "mfcc3_mean",
    "mfcc4_mean":                  "mfcc4_mean",
    "mfcc5_mean":                  "mfcc5_mean",
    "mfcc6_mean":                  "mfcc6_mean",
    "mfcc7_mean":                  "mfcc7_mean",
    "mfcc8_mean":                  "mfcc8_mean",
    "mfcc9_mean":                  "mfcc9_mean",
    "mfcc10_mean":                 "mfcc10_mean",
    "mfcc11_mean":                 "mfcc11_mean",
    "mfcc12_mean":                 "mfcc12_mean",
    "mfcc13_mean":                 "mfcc13_mean",
    "spectral_centroid_mean":      "spectral_centroid_mean",
    "spectral_bandwidth_mean":     "spectral_bandwidth_mean",
    "rolloff_mean":                "rolloff_mean",
    "zero_crossing_rate_mean":     "zero_crossing_rate_mean",
    "zero_crossing_rate_var":      "zero_crossing_rate_var",
    "chroma_stft_mean":            "chroma_stft_mean",
    "chroma_stft_var":             "chroma_stft_var",
    "rmse_mean":                   "rmse_mean",
    "rmse_var":                    "rmse_var",
    "harmony_mean":                "harmony_mean",
    "harmony_var":                 "harmony_var",
    "tempo":                       "tempo",
    "beats":                       "beats",
}

# ── Train one XGBoost model per stem type ─────────────────────
stem_accuracies = {}

for stem_type, feature_list in STEM_FEATURE_GROUPS.items():
    print(f"\n── Training classifier for: {stem_type} ──")

    # Only use columns that exist in the GTZAN dataset
    available = [f for f in feature_list if f in X.columns]
    missing   = [f for f in feature_list if f not in X.columns]

    if missing:
        print(f"  Note: {len(missing)} features not in GTZAN, skipping: {missing[:3]}...")

    if len(available) < 3:
        print(f"  Not enough features — using all columns instead")
        available = list(X.columns)

    X_stem = X[available]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_stem)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42
    )

    # XGBoost classifier
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mlogloss",
        verbosity=0
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    stem_accuracies[stem_type] = round(acc * 100, 1)
    print(f"  Accuracy: {acc*100:.1f}%")

    # Cross validation
    cv_scores = cross_val_score(model, X_scaled, y_encoded, cv=5, scoring="accuracy")
    print(f"  CV Score: {cv_scores.mean()*100:.1f}% (+/- {cv_scores.std()*100:.1f}%)")

    # Save model, scaler, feature list for this stem
    joblib.dump(model,     f"model/{stem_type}_model.pkl")
    joblib.dump(scaler,    f"model/{stem_type}_scaler.pkl")
    joblib.dump(available, f"model/{stem_type}_features.pkl")

# ── Save shared label encoder ─────────────────────────────────
joblib.dump(le, "model/label_encoder.pkl")

# ── Save accuracy summary ─────────────────────────────────────
with open("model/accuracies.json", "w") as f:
    json.dump(stem_accuracies, f, indent=2)

print("\n=== Summary ===")
for stem, acc in stem_accuracies.items():
    print(f"  {stem:10s}: {acc}%")
print("\nAll models saved to model/")