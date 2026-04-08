# features.py
import librosa
import numpy as np
import io

def extract_features(audio_bytes: bytes) -> dict:
    """
    Extracts audio features from a stem's bytes.
    Returns a dict of all features for use in dashboard, classifier, and report.
    """
    if audio_bytes is None:
        return None

    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)

    # ── Skip silent/empty stems ───────────────────────────────
    if np.max(np.abs(y)) < 0.01:
        return None

    features = {}

    # ── Tempo & Rhythm ────────────────────────────────────────
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    features["tempo"]         = round(float(tempo), 2)
    features["beat_count"]    = int(len(beats))

    # ── Energy & Loudness ─────────────────────────────────────
    rms = librosa.feature.rms(y=y)[0]
    features["energy_mean"]   = round(float(np.mean(rms)), 4)
    features["energy_max"]    = round(float(np.max(rms)), 4)
    features["energy_std"]    = round(float(np.std(rms)), 4)

    # ── Spectral Features ─────────────────────────────────────
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]

    features["spectral_centroid_mean"]  = round(float(np.mean(spectral_centroids)), 2)
    features["spectral_rolloff_mean"]   = round(float(np.mean(spectral_rolloff)), 2)
    features["spectral_bandwidth_mean"] = round(float(np.mean(spectral_bandwidth)), 2)

    # ── Zero Crossing Rate ────────────────────────────────────
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    features["zcr_mean"]      = round(float(np.mean(zcr)), 4)

    # ── MFCCs (13 coefficients) ───────────────────────────────
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i, mfcc in enumerate(mfccs):
        features[f"mfcc_{i+1}"] = round(float(np.mean(mfcc)), 4)

    # ── Chroma (harmonic content) ─────────────────────────────
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features["chroma_mean"]   = round(float(np.mean(chroma)), 4)
    features["chroma_std"]    = round(float(np.std(chroma)), 4)

    # ── Silence Ratio ─────────────────────────────────────────
    silence_threshold = 0.01
    silent_frames = np.sum(np.abs(y) < silence_threshold)
    features["silence_ratio"] = round(float(silent_frames / len(y)), 4)

    # ── Duration ──────────────────────────────────────────────
    features["duration_sec"]  = round(float(librosa.get_duration(y=y, sr=sr)), 2)

    return features


def extract_all_stems(stem_data: dict) -> dict:
    """
    Runs extract_features on every stem.
    Returns a dict like { "vocals": {...}, "drums": {...}, ... }
    """
    all_features = {}
    for stem_key, audio_bytes in stem_data.items():
        print(f"Extracting features for: {stem_key}")
        all_features[stem_key] = extract_features(audio_bytes)
    return all_features