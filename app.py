# app.py
import streamlit as st
import tempfile
import os
import io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import librosa

from separator import separate_stems
from features import extract_all_stems
from classifier import predict_all_stems
from reporter import generate_report, generate_csv

st.set_page_config(
    page_title="AI Stem Separator",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 AI Stem Separator & Music Intelligence")
st.write("Upload a song to separate stems, analyze audio features, classify genres, and generate a full report.")

STEM_META = {
    "vocals":  {"icon": "🎤", "label": "Vocals"},
    "drums":   {"icon": "🥁", "label": "Drums"},
    "bass":    {"icon": "🎸", "label": "Bass"},
    "guitar":  {"icon": "🎸", "label": "Guitar"},
    "piano":   {"icon": "🎹", "label": "Piano"},
    "other":   {"icon": "🎼", "label": "Other"},
}

WAVEFORM_COLORS = {
    "vocals": "#4fc3f7",
    "drums":  "#ef5350",
    "bass":   "#66bb6a",
    "guitar": "#ffa726",
    "piano":  "#ab47bc",
    "other":  "#78909c",
}

GENRE_COLORS = {
    "classical": "#7986CB", "jazz": "#4DB6AC",
    "blues": "#64B5F6",     "rock": "#E57373",
    "metal": "#90A4AE",     "pop": "#F06292",
    "hiphop": "#FFB74D",    "country": "#A5D6A7",
    "disco": "#CE93D8",     "reggae": "#80CBC4",
}

# ── Session state ─────────────────────────────────────────────
for key in ["stem_data", "song_name", "all_features", "all_predictions"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ── Sidebar — upload & separate ───────────────────────────────
with st.sidebar:
    st.header("Upload Song")
    uploaded_file = st.file_uploader(
        "Choose an MP3 or WAV file",
        type=["mp3", "wav"],
    )

    if uploaded_file:
        st.audio(uploaded_file, format="audio/mp3")
        st.info(f"📄 `{uploaded_file.name}`")

        if st.button("🚀 Separate & Analyze", use_container_width=True):
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=os.path.splitext(uploaded_file.name)[1]
            ) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            # ── Step 1: Separate ──────────────────────────────
            with st.spinner("Step 1/3 — Separating stems..."):
                try:
                    stems = separate_stems(tmp_path)
                    stem_data = {}
                    for stem_key, stem_path in stems.items():
                        if stem_path and os.path.exists(stem_path):
                            with open(stem_path, "rb") as f:
                                stem_data[stem_key] = f.read()
                        else:
                            stem_data[stem_key] = None
                    st.session_state.stem_data = stem_data
                    st.session_state.song_name = os.path.splitext(uploaded_file.name)[0]
                except Exception as e:
                    st.error(f"Separation failed: {e}")
                    st.stop()
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

            # ── Step 2: Extract features ──────────────────────
            with st.spinner("Step 2/3 — Extracting audio features..."):
                try:
                    st.session_state.all_features = extract_all_stems(
                        st.session_state.stem_data
                    )
                except Exception as e:
                    st.error(f"Feature extraction failed: {e}")
                    st.stop()

            # ── Step 3: Classify genres ───────────────────────
            with st.spinner("Step 3/3 — Classifying genres..."):
                try:
                    st.session_state.all_predictions = predict_all_stems(
                        st.session_state.all_features
                    )
                except Exception as e:
                    st.error(f"Classification failed: {e}")
                    st.stop()

            st.success("✅ Analysis complete!")

    # ── Sidebar status ────────────────────────────────────────
    if st.session_state.song_name:
        st.divider()
        st.markdown(f"**Current song:**")
        st.markdown(f"`{st.session_state.song_name}`")
        detected = [
            k for k, v in st.session_state.stem_data.items()
            if v is not None
        ] if st.session_state.stem_data else []
        st.markdown(f"**Detected stems:** {len(detected)}/6")
        for s in detected:
            st.markdown(f"  {STEM_META[s]['icon']} {STEM_META[s]['label']}")

# ── Main area — tabs ──────────────────────────────────────────
if st.session_state.stem_data is None:
    st.info("👈 Upload a song in the sidebar to get started.")
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎵 Stems",
    "📊 Analytics Dashboard",
    "🎯 Genre Classification",
    "📄 Report",
    "🧠 Model Performance"
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — Stems
# ══════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Separated Stems")
    st.info(
        "ℹ️ Stem separation is powered by Meta's Demucs `htdemucs_6s` model. "
        "Results are generally accurate but may vary on complex mixes — "
        "instruments sharing similar frequency ranges (e.g. piano & guitar) "
        "can occasionally be misassigned. This is a known open-source limitation."
    )
    cols = st.columns(2)

    for i, (stem_key, meta) in enumerate(STEM_META.items()):
        audio_bytes = st.session_state.stem_data.get(stem_key)
        col = cols[i % 2]
        with col:
            st.markdown(f"**{meta['icon']} {meta['label']}**")
            if audio_bytes:
                st.audio(audio_bytes, format="audio/mp3")
                col.download_button(
                    label=f"⬇️ Download {meta['label']}",
                    data=audio_bytes,
                    file_name=f"{st.session_state.song_name}_{stem_key}.mp3",
                    mime="audio/mpeg",
                    use_container_width=True,
                    key=f"dl_{stem_key}"
                )
            else:
                col.info("〰️ Not detected in this song")

    # ── Waveforms ─────────────────────────────────────────────
    st.divider()
    st.subheader("📊 Waveform Comparison")

    if st.session_state.stem_data:
        fig, axes = plt.subplots(
            len(STEM_META), 1,
            figsize=(10, 2 * len(STEM_META))
        )
        fig.patch.set_facecolor('#0e1117')

        for ax, (stem_key, meta) in zip(axes, STEM_META.items()):
            audio_bytes = st.session_state.stem_data.get(stem_key)
            ax.set_facecolor('#0e1117')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_edgecolor('#333')

            if audio_bytes:
                y, sr = librosa.load(
                    io.BytesIO(audio_bytes),
                    sr=None, mono=True, duration=30
                )
                times = np.linspace(0, len(y) / sr, num=len(y))
                ax.plot(
                    times, y,
                    color=WAVEFORM_COLORS.get(stem_key, "#aaa"),
                    linewidth=0.4, alpha=0.85
                )
                ax.set_ylabel(meta['label'], color='white', fontsize=9)
                ax.set_ylim(-1, 1)
            else:
                ax.text(
                    0.5, 0.5, f"{meta['label']} — not detected",
                    ha='center', va='center',
                    color='#666', fontsize=10,
                    transform=ax.transAxes
                )
                ax.set_ylabel(meta['label'], color='#555', fontsize=9)

        axes[-1].set_xlabel("Time (seconds)", color='white', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ══════════════════════════════════════════════════════════════
# TAB 2 — Analytics Dashboard
# ══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Audio Analytics Dashboard")

    if not st.session_state.all_features:
        st.info("No features extracted yet.")
        st.stop()

    detected_stems = {
        k: v for k, v in st.session_state.all_features.items()
        if v is not None
    }

    if not detected_stems:
        st.warning("No stems were detected in this song.")
        st.stop()

    # ── Energy comparison bar chart ───────────────────────────
    st.markdown("#### Energy by Stem")
    st.caption(
        "RMS energy measures how loud or prominent each stem is in the mix. "
        "Higher = more dominant. Vocals and drums typically score highest in most songs. "
        "A flat or near-zero bar means that instrument was barely present."
    )
    st.markdown(
        "[Learn more about RMS energy in audio](https://en.wikipedia.org/wiki/Root_mean_square#In_audio)",
        unsafe_allow_html=False
    )
    energy_data = {
        STEM_META[s]["label"]: v["energy_mean"]
        for s, v in detected_stems.items()
    }
    fig1, ax1 = plt.subplots(figsize=(8, 3))
    fig1.patch.set_facecolor('#0e1117')
    ax1.set_facecolor('#0e1117')
    bars = ax1.bar(
        energy_data.keys(),
        energy_data.values(),
        color=[WAVEFORM_COLORS[s] for s in detected_stems.keys()],
        edgecolor='none'
    )
    ax1.tick_params(colors='white')
    ax1.set_ylabel("Mean Energy (RMS)", color='white', fontsize=9)
    for spine in ax1.spines.values():
        spine.set_edgecolor('#333')
    for bar, val in zip(bars, energy_data.values()):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{val:.4f}", ha='center', va='bottom',
            color='white', fontsize=8
        )
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

    # ── Spectral features comparison ──────────────────────────
    st.markdown("#### Spectral Profile by Stem")
    st.caption(
        "These three frequency-based measurements describe the tonal character of each stem. "
        "Centroid = brightness (high = treble-heavy, low = bass-heavy). "
        "Rolloff = where most of the energy is concentrated. "
        "Bandwidth = how wide the frequency range is — wide means complex, narrow means pure."
    )
    st.markdown(
        "[Learn more about spectral features](https://librosa.org/doc/latest/feature.html#spectral-features)"
    )
    spec_metrics = ["spectral_centroid_mean", "spectral_rolloff_mean", "spectral_bandwidth_mean"]
    spec_labels  = ["Centroid", "Rolloff", "Bandwidth"]

    spec_data = {
        STEM_META[s]["label"]: [v[m] for m in spec_metrics]
        for s, v in detected_stems.items()
    }
    df_spec = pd.DataFrame(spec_data, index=spec_labels)

    fig2, ax2 = plt.subplots(figsize=(8, 3))
    fig2.patch.set_facecolor('#0e1117')
    ax2.set_facecolor('#0e1117')
    df_spec.T.plot(
        kind='bar', ax=ax2,
        color=[WAVEFORM_COLORS[s] for s in detected_stems.keys()],
        edgecolor='none'
    )
    ax2.tick_params(colors='white', axis='both')
    ax2.set_ylabel("Hz", color='white', fontsize=9)
    ax2.legend(
        spec_labels, fontsize=8,
        facecolor='#1a1a2e', labelcolor='white',
        loc='upper right'
    )
    for spine in ax2.spines.values():
        spine.set_edgecolor('#333')
    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    # ── MFCC heatmap ──────────────────────────────────────────
    st.markdown("#### MFCC Heatmap (Timbral Fingerprint)")
    st.caption(
        "MFCCs (Mel-Frequency Cepstral Coefficients) capture the texture and timbre of each stem — "
        "think of it as a sonic fingerprint. Each row is a coefficient (MFCC 1–13), each column is a stem. "
        "Red = high positive values, blue = high negative. "
        "Similar columns mean the stems sound tonally similar. "
        "This is also what the genre classifier uses under the hood."
    )
    st.markdown(
        "[Learn more about MFCCs](https://librosa.org/doc/latest/feature.html#mfcc)"
    )
    mfcc_cols = [f"mfcc_{i}" for i in range(1, 14)]
    mfcc_data = {
        STEM_META[s]["label"]: [v.get(m, 0) for m in mfcc_cols]
        for s, v in detected_stems.items()
    }
    df_mfcc = pd.DataFrame(mfcc_data, index=[f"MFCC {i}" for i in range(1, 14)])

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    fig3.patch.set_facecolor('#0e1117')
    sns.heatmap(
        df_mfcc, ax=ax3, cmap="coolwarm",
        annot=True, fmt=".0f", annot_kws={"size": 7},
        linewidths=0.3, linecolor='#333',
        cbar_kws={"shrink": 0.8}
    )
    
    ax3.tick_params(axis='x', colors='white', labelsize=9)
    ax3.tick_params(axis='y', colors='white', labelsize=8)
    
    cbar = ax3.collections[0].colorbar
    ax3.tick_params(colors='white', labelsize=7)
    cbar.ax.yaxis.label.set_color('white')
    cbar.outline.set_edgecolor('#555')
    ax3.set_facecolor('#0e1117')
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    # ── Feature summary table ─────────────────────────────────
    st.markdown("#### Full Feature Table")
    st.caption(
        "A summary of all key metrics per stem. "
        "Tempo = estimated BPM. Brightness = spectral centroid in Hz. "
        "Percussiveness = zero crossing rate (how often the signal changes sign — high in drums). "
        "Silence % = how much of the stem is quiet."
    )
    rows = []
    for stem, feats in detected_stems.items():
        rows.append({
            "Stem":          STEM_META[stem]["label"],
            "Tempo (BPM)":   feats.get("tempo"),
            "Energy":        feats.get("energy_mean"),
            "Brightness":    feats.get("spectral_centroid_mean"),
            "Percussiveness":feats.get("zcr_mean"),
            "Silence %":     f"{round(feats.get('silence_ratio', 0) * 100, 1)}%",
            "Duration (s)":  feats.get("duration_sec"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════
# TAB 3 — Genre Classification
# ══════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Genre Classification")
    st.caption("Each stem is classified independently using a Random Forest model trained on the GTZAN dataset.")

    if not st.session_state.all_predictions:
        st.info("No predictions yet.")
        st.stop()

    detected_preds = {
        k: v for k, v in st.session_state.all_predictions.items()
        if v is not None
    }

    if not detected_preds:
        st.warning("No stems detected to classify.")
        st.stop()

    # ── Top genre per stem ────────────────────────────────────
    cols = st.columns(len(detected_preds))
    for col, (stem, pred) in zip(cols, detected_preds.items()):
        genre = pred["top_genre"]
        conf  = pred["confidence"]
        inconclusive = pred.get("inconclusive", False)

        # Gray out inconclusive results
        color = "#888888" if inconclusive else GENRE_COLORS.get(genre, "#aaaaaa")

        col.markdown(
            f"""
            <div style='background:{color}22; border:1px solid {color};
                        border-radius:8px; padding:12px; text-align:center;'>
                <div style='font-size:13px; color:#aaa;'>
                    {STEM_META[stem]['icon']} {STEM_META[stem]['label']}
                </div>
                <div style='font-size:18px; font-weight:600; color:{color};'>
                    {genre.title()}
                </div>
                <div style='font-size:11px; color:#aaa;'>{conf}% confidence</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if inconclusive and pred.get("note"):
            col.caption(pred["note"])

    # ── Genre probability breakdown ───────────────────────────
    st.markdown("#### Full Genre Probability Breakdown")
    for stem, pred in detected_preds.items():
        st.markdown(f"**{STEM_META[stem]['icon']} {STEM_META[stem]['label']}**")
        all_genres = pred.get("all_genres", {})
        fig, ax = plt.subplots(figsize=(8, 2))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        genre_names = list(all_genres.keys())
        genre_vals  = list(all_genres.values())
        bar_colors  = [GENRE_COLORS.get(g, "#aaa") for g in genre_names]
        ax.barh(genre_names, genre_vals, color=bar_colors, edgecolor='none')
        ax.tick_params(colors='white')
        ax.set_xlabel("Confidence %", color='white', fontsize=8)
        ax.set_xlim(0, 100)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333')
        for i, val in enumerate(genre_vals):
            if val > 2:
                ax.text(val + 0.5, i, f"{val}%", va='center',
                        color='white', fontsize=7)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Model transparency note ───────────────────────────────
    st.divider()
    with st.expander("ℹ️ About the genre classifier"):
        st.markdown("""
        **How it works:**
        The genre classifier uses a separate XGBoost model per stem type,
        trained on the GTZAN dataset (1,000 songs across 10 genres).

        **Known limitation:**
        The GTZAN dataset contains full mixed songs, not isolated stems.
        This means the model is predicting genre from a stem that sounds
        very different from what it was trained on — confidence scores
        below 55% are marked as **Inconclusive** rather than showing a
        potentially misleading result.

        **What this means for you:**
        Inconclusive results aren't failures — they're honest signals that
        the stem's audio profile doesn't strongly match any single genre
        in the training data. A future improvement would retrain on a
        stem-specific labeled dataset.
        """)

# ══════════════════════════════════════════════════════════════
# TAB 4 — Report
# ══════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Stem Intelligence Report")
    st.write("Download a full analysis report of this song as PDF or CSV.")

    if not st.session_state.all_features or not st.session_state.all_predictions:
        st.info("Separate and analyze a song first to generate a report.")
        st.stop()

    col1, col2 = st.columns(2)

    # ── PDF download ──────────────────────────────────────────
    with col1:
        st.markdown("#### PDF Report")
        st.caption("Full formatted report with summary, genre classifications, audio features, and insights.")
        if st.button("📄 Generate PDF", use_container_width=True):
            with st.spinner("Generating PDF..."):
                try:
                    pdf_bytes = generate_report(
                        st.session_state.song_name,
                        st.session_state.all_features,
                        st.session_state.all_predictions
                    )
                    st.download_button(
                        label="⬇️ Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"{st.session_state.song_name}_stem_report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        key="dl_pdf"
                    )
                    st.success("PDF ready!")
                except Exception as e:
                    st.error(f"PDF generation failed: {e}")

    # ── CSV download ──────────────────────────────────────────
    with col2:
        st.markdown("#### CSV Data Export")
        st.caption("Raw feature data and genre predictions for every stem — ready for Excel or further analysis.")
        if st.button("📊 Generate CSV", use_container_width=True):
            with st.spinner("Generating CSV..."):
                try:
                    csv_bytes = generate_csv(
                        st.session_state.song_name,
                        st.session_state.all_features,
                        st.session_state.all_predictions
                    )
                    st.download_button(
                        label="⬇️ Download CSV",
                        data=csv_bytes,
                        file_name=f"{st.session_state.song_name}_stem_data.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="dl_csv"
                    )
                    st.success("CSV ready!")
                except Exception as e:
                    st.error(f"CSV generation failed: {e}")

    # ── Preview insights inline ───────────────────────────────
    st.divider()
    st.markdown("#### Key Insights Preview")
    from reporter import generate_insights
    insights = generate_insights(
        [k for k, v in st.session_state.all_features.items() if v],
        st.session_state.all_features,
        st.session_state.all_predictions
    )
    for insight in insights:
        st.markdown(f"• {insight}")

# ══════════════════════════════════════════════════════════════
# TAB 5 — Model Performance
# ══════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Model Performance — Confusion Matrix")
    st.caption(
        "How well the genre classifier performs on the GTZAN test set. "
        "Each row = actual genre, each column = predicted genre. "
        "A perfect model would have all values on the diagonal."
    )

    from classifier import get_confusion_matrix_data
    import json

    with st.spinner("Loading model performance data..."):
        cm_data = get_confusion_matrix_data()

    if cm_data is None:
        st.warning("features_30_sec.csv not found — place it in your project folder to view model performance.")
    else:
        # ── Accuracy metric ───────────────────────────────────
        st.metric(
            label="Overall Classifier Accuracy (GTZAN test set)",
            value=f"{cm_data['accuracy']}%",
            delta="vs 10% random baseline",
        )

        st.divider()

        # ── Confusion matrix heatmap ──────────────────────────
        cm_array  = np.array(cm_data["matrix"])
        classes   = cm_data["classes"]

        # Normalize to percentages per row
        cm_norm = cm_array.astype(float)
        row_sums = cm_norm.sum(axis=1, keepdims=True)
        cm_norm  = np.divide(cm_norm, row_sums, where=row_sums != 0) * 100

        fig, ax = plt.subplots(figsize=(9, 7))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')

        sns.heatmap(
            cm_norm, ax=ax,
            annot=True, fmt=".0f",
            cmap="Blues",
            xticklabels=[c.title() for c in classes],
            yticklabels=[c.title() for c in classes],
            annot_kws={"size": 9},
            linewidths=0.3, linecolor='#222',
            cbar_kws={"shrink": 0.8}
        )

        ax.set_xlabel("Predicted Genre", color='white', fontsize=10, labelpad=10)
        ax.set_ylabel("Actual Genre",    color='white', fontsize=10, labelpad=10)
        ax.tick_params(axis='x', colors='white', labelsize=9, rotation=30)
        ax.tick_params(axis='y', colors='white', labelsize=9, rotation=0)

        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(colors='white', labelsize=8)
        cbar.ax.yaxis.label.set_color('white')
        cbar.outline.set_edgecolor('#555')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # ── Per genre accuracy breakdown ──────────────────────
        st.divider()
        st.markdown("#### Per-Genre Accuracy")
        st.caption("Diagonal values from the matrix above — how often each genre was correctly identified.")

        per_genre = {
            classes[i]: round(cm_norm[i][i], 1)
            for i in range(len(classes))
        }
        sorted_genres = sorted(per_genre.items(), key=lambda x: x[1], reverse=True)

        fig2, ax2 = plt.subplots(figsize=(8, 3))
        fig2.patch.set_facecolor('#0e1117')
        ax2.set_facecolor('#0e1117')

        genre_names = [g.title() for g, _ in sorted_genres]
        genre_vals  = [v for _, v in sorted_genres]
        bar_colors  = [GENRE_COLORS.get(g, "#aaa") for g, _ in sorted_genres]

        bars = ax2.bar(genre_names, genre_vals, color=bar_colors, edgecolor='none')
        ax2.tick_params(colors='white', labelsize=9)
        ax2.set_ylabel("Accuracy %", color='white', fontsize=9)
        ax2.set_ylim(0, 110)
        for spine in ax2.spines.values():
            spine.set_edgecolor('#333')
        for bar, val in zip(bars, genre_vals):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{val}%", ha='center', color='white', fontsize=8
            )
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        # ── Insight callouts ──────────────────────────────────
        st.divider()
        best  = max(per_genre, key=per_genre.get)
        worst = min(per_genre, key=per_genre.get)
        col1, col2 = st.columns(2)
        col1.success(f"✅ Best: **{best.title()}** at {per_genre[best]}% accuracy")
        col2.error(f"⚠️ Hardest: **{worst.title()}** at {per_genre[worst]}% accuracy")

        st.caption(
            "Low accuracy on certain genres (e.g. Rock, Reggae) reflects overlapping "
            "audio characteristics in the GTZAN dataset — not a bug, but a known "
            "challenge in music genre classification research."
        )