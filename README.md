---
title: AI Stem Separator
emoji: 🎵
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.56.0
app_file: app.py
pinned: false
---

# 🎵 AI Stem Separator & Music Intelligence

An end-to-end audio analysis tool that separates any song into individual instrument stems, analyses each one, classifies genres, and generates downloadable reports — all from a single upload.

Built as a data science portfolio project by someone who used to make acapellas by ear. What once took days of listening and recording now takes minutes.

---

## What It Does

Upload any MP3 or WAV file and the app will:

- **Separate** the song into up to 6 individual stems — vocals, drums, bass, guitar, piano, and other
- **Visualise** the waveform and energy profile of each stem
- **Analyse** 25+ audio features per stem including tempo, brightness, timbral texture, and silence ratio
- **Classify** the genre of each stem independently using a machine learning model
- **Compare** two songs side by side across every stem and feature
- **Generate** a full PDF intelligence report and CSV data export

---

## Live App

🔗 https://anirudh-rs-ai-stem-separator.hf.space

---

## Features by Tab

| Tab | What It Shows |
|---|---|
| 🎵 Stems | Separated stems with audio preview and individual downloads |
| 📊 Analytics Dashboard | Energy charts, spectral profile, MFCC heatmap, feature table |
| 🎯 Genre Classification | Per-stem genre prediction with confidence scores |
| 📄 Report | PDF and CSV download for the full analysis |
| 🧠 Model Performance | Confusion matrix and per-genre accuracy breakdown |
| 🔀 Compare Songs | Side by side waveform, feature, and genre comparison for two songs |

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Meta Demucs (`htdemucs_6s`) | AI stem separation |
| librosa | Audio feature extraction |
| XGBoost + scikit-learn | Per-stem genre classification |
| Streamlit | Web interface |
| GTZAN Dataset | Model training data (1,000 songs, 10 genres) |
| reportlab | PDF report generation |
| Hugging Face Spaces | Cloud deployment |

---

## How the Genre Classifier Works

Each stem type (vocals, drums, bass, guitar, piano, other) has its own XGBoost model trained on the GTZAN dataset. Features most relevant to each instrument are used — for example, drums use zero crossing rate and beat count, while piano and guitar use chroma and harmonic features.

Predictions below 55% confidence are shown as **Inconclusive** rather than displaying a potentially misleading result.

---

## Known Limitations

These are documented transparently in the app itself:

- **Guitar and piano misassignment** — both instruments share overlapping frequency ranges, particularly in dense mixes. Demucs occasionally assigns piano frequencies to the guitar stem or vice versa
- **Genre classifier trained on full songs** — the GTZAN dataset contains complete mixed recordings, not isolated stems. This creates a mismatch when classifying single-instrument stems, which is why confidence thresholding is applied
- **Complex productions** — heavily layered recordings like Queen or classic rock wall-of-sound mixes are harder to separate cleanly than acoustic or sparse arrangements
- **Separation quality varies** — open source stem separation is impressive but not perfect. Professional studio multitrack files would always produce cleaner results

---

## Results on Test Songs

| Song | Vocals | Drums | Bass | Guitar | Piano | Notes |
|---|---|---|---|---|---|---|
| Blackbird — The Beatles | ✅ | ✅ | ❌ | ✅ | ❌ | Acoustic only — correct |
| Someone Like You — Adele | ✅ | ❌ | ❌ | ❌ | ✅ | Piano ballad — correct |
| Beat It — Michael Jackson | ✅ | ✅ | ✅ | ✅ | ❌ | Rock/pop — correct |
| Bohemian Rhapsody — Queen | ✅ | ✅ | ✅ | ⚠️ | ⚠️ | Dense mix — partial misassignment |

---

## Project Structure

```
ai-stem-separator/
├── app.py                    ← Main Streamlit application
├── separator.py              ← Demucs stem separation logic
├── features.py               ← Audio feature extraction via librosa
├── classifier.py             ← Per-stem genre classification
├── reporter.py               ← PDF and CSV report generation
├── train_model.py            ← Original single-model training script
├── train_stem_classifier.py  ← Per-stem XGBoost training script
├── requirements.txt          ← Python dependencies
├── packages.txt              ← System dependencies (ffmpeg)
├── .streamlit/
│   └── config.toml           ← Streamlit configuration
└── model/                    ← Trained model files (auto-generated on first run)
```

---

## Running Locally

```bash
# Create environment
conda create -n stem-tool python=3.10
conda activate stem-tool

# Install dependencies
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Train models (requires features_30_sec.csv from GTZAN dataset)
python train_stem_classifier.py

# Run app
streamlit run app.py
```

---

## Business Applications

This kind of audio intelligence tooling sits at the core of several real industries:

- **Music education** — isolate any instrument from any song for teaching
- **Music production** — analyse how commercial tracks are mixed and mastered
- **Streaming platforms** — feature extraction powers recommendation engines
- **Sync licensing** — quickly assess whether a track fits a brief
- **A&R scouting** — compare new demos against established artists sonically

---

## What This Project Demonstrates

- End-to-end ML pipeline on unstructured data (raw audio → deployed web app)
- Feature engineering from real-world files using domain knowledge
- Model training, evaluation, and honest performance reporting
- Confidence thresholding and transparent handling of out-of-distribution predictions
- Full cloud deployment with automated first-run model training

---

## Acknowledgements

- [Meta Demucs](https://github.com/facebookresearch/demucs) — open source stem separation
- [GTZAN Dataset](https://marsyas.info/downloads/data_sets.html) — music genre classification benchmark
- [librosa](https://librosa.org) — audio analysis library

---

*Built with Python · Streamlit · Meta Demucs · librosa · XGBoost · Hugging Face Spaces*
