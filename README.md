# AI Stem Separator & Music Intelligence

An end-to-end audio analysis tool that separates songs into individual stems,
analyzes audio features, classifies genres, and generates intelligence reports.

## Features
- 6-stem separation (vocals, drums, bass, guitar, piano, other)
- Audio analytics dashboard with waveform visualization
- Genre classification with confidence thresholding
- MFCC heatmap and spectral analysis
- Confusion matrix and model performance tab
- PDF and CSV report generation

## Tech Stack
- Meta Demucs (stem separation)
- librosa (audio feature extraction)
- XGBoost + scikit-learn (genre classification)
- Streamlit (UI)
- GTZAN dataset (model training)

## Known Limitations
- Guitar/piano misassignment on dense mixes (frequency overlap)
- Genre classifier trained on full songs, not isolated stems
- Confidence threshold of 55% applied — low confidence shown as Inconclusive

## Demo
- The repository file contains a demo video showing the stem functionality
