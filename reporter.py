# reporter.py
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import io
import os
from datetime import datetime

STEM_ICONS = {
    "vocals": "Vocals", "drums": "Drums", "bass": "Bass",
    "guitar": "Guitar", "piano": "Piano", "other": "Other",
}

PAGE_W = A4[0] - 5*cm  # usable width at 2.5cm margins each side

def _base_table_style(header_font_size=9, extra=None):
    style = [
        ("BACKGROUND",     (0, 0), (-1, 0),  colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR",      (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",       (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, 0),  header_font_size),
        ("FONTSIZE",       (0, 1), (-1, -1), 8),
        ("FONTNAME",       (0, 1), (-1, -1), "Helvetica"),
        ("TEXTCOLOR",      (0, 1), (-1, -1), colors.HexColor("#333333")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#f4f4f4"), colors.HexColor("#ffffff")]),
        ("GRID",           (0, 0), (-1, -1), 0.4, colors.HexColor("#dddddd")),
        ("TOPPADDING",     (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
        ("LEFTPADDING",    (0, 0), (-1, -1), 7),
        ("RIGHTPADDING",   (0, 0), (-1, -1), 7),
        ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
    ]
    if extra:
        style.extend(extra)
    return style


def generate_report(song_name, all_features, all_predictions, output_path=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=2.5*cm, leftMargin=2.5*cm,
        topMargin=2*cm, bottomMargin=2*cm
    )

    styles = getSampleStyleSheet()

    section_style = ParagraphStyle(
        "section", parent=styles["Normal"],
        fontSize=11, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#1a1a2e"),
        spaceBefore=10, spaceAfter=5, leading=14
    )
    body_style = ParagraphStyle(
        "body", parent=styles["Normal"],
        fontSize=8, fontName="Helvetica",
        textColor=colors.HexColor("#444444"),
        spaceAfter=4, leading=12
    )
    small_style = ParagraphStyle(
        "small", parent=styles["Normal"],
        fontSize=7, fontName="Helvetica",
        textColor=colors.HexColor("#999999"),
        spaceAfter=0, alignment=TA_CENTER, leading=10
    )

    story = []

    # ── Header banner ─────────────────────────────────────────
    title_style = ParagraphStyle(
        "title_dark", parent=styles["Normal"],
        fontSize=18, fontName="Helvetica-Bold",
        textColor=colors.white, alignment=TA_CENTER, leading=22, spaceAfter=2
    )
    meta_style = ParagraphStyle(
        "meta_dark", parent=styles["Normal"],
        fontSize=8, fontName="Helvetica",
        textColor=colors.HexColor("#aaaaaa"), alignment=TA_CENTER, leading=11
    )
    header_data = [
        [Paragraph("Stem Intelligence Report", title_style)],
        [Paragraph(
            f"{song_name}  ·  Generated {datetime.now().strftime('%B %d, %Y  %H:%M')}",
            meta_style
        )]
    ]
    header_table = Table(header_data, colWidths=[PAGE_W])
    header_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#1a1a2e")),
        ("TOPPADDING",    (0, 0), (0, 0),   10),
        ("BOTTOMPADDING", (0, 0), (0, 0),   2),
        ("TOPPADDING",    (0, 1), (0, 1),   0),
        ("BOTTOMPADDING", (0, 1), (0, 1),   10),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 0.15*cm))

    # ── 1. Song Summary ───────────────────────────────────────
    detected     = [s for s, f in all_features.items() if f is not None]
    not_detected = [s for s, f in all_features.items() if f is None]

    duration = "N/A"
    for stem in detected:
        if all_features[stem] and "duration_sec" in all_features[stem]:
            secs = int(all_features[stem]["duration_sec"])
            duration = f"{secs // 60}m {secs % 60}s"
            break

    story.append(Paragraph("1. Song Summary", section_style))
    summary_data = [
        ["Property", "Value"],
        ["Song Name",            song_name],
        ["Duration",             duration],
        ["Stems Detected",       ", ".join(detected) if detected else "None"],
        ["Stems Not Detected",   ", ".join(not_detected) if not_detected else "None"],
        ["Total Stems Analyzed", f"{len(detected)} of 6"],
    ]
    col_a = 4*cm
    col_b = PAGE_W - col_a
    summary_table = Table(summary_data, colWidths=[col_a, col_b])
    summary_table.setStyle(TableStyle(_base_table_style(extra=[
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
    ])))
    story.append(summary_table)

    # ── 2. Genre Classification ───────────────────────────────
    story.append(Paragraph("2. Genre Classification", section_style))
    story.append(Paragraph(
        "Each stem analyzed independently via Random Forest classifier — GTZAN dataset (1,000 songs, 10 genres).",
        body_style
    ))
    genre_data = [["Stem", "Top Genre", "Confidence", "2nd Genre", "3rd Genre"]]
    for stem in detected:
        pred = all_predictions.get(stem)
        if pred and pred.get("top_3"):
            t = pred["top_3"]
            genre_data.append([
                STEM_ICONS.get(stem, stem),
                t[0]["genre"].title() if len(t) > 0 else "—",
                f"{t[0]['confidence']}%"  if len(t) > 0 else "—",
                t[1]["genre"].title() if len(t) > 1 else "—",
                t[2]["genre"].title() if len(t) > 2 else "—",
            ])
    if len(genre_data) > 1:
        g1=2.8*cm; g2=3.2*cm; g3=2.6*cm; g4=3.2*cm; g5=PAGE_W-g1-g2-g3-g4
        genre_table = Table(genre_data, colWidths=[g1, g2, g3, g4, g5])
        genre_table.setStyle(TableStyle(_base_table_style(extra=[
            ("ALIGN", (2, 1), (2, -1), "CENTER"),
        ])))
        story.append(genre_table)

    # ── 3. Audio Feature Analysis ─────────────────────────────
    story.append(Paragraph("3. Audio Feature Analysis", section_style))
    story.append(Paragraph(
        "Key acoustic properties extracted per stem using librosa.",
        body_style
    ))
    feature_data = [["Stem", "BPM", "Energy", "Brightness", "Percussive", "Silence%", "Dur(s)"]]
    for stem in detected:
        f = all_features[stem]
        if f:
            feature_data.append([
                STEM_ICONS.get(stem, stem),
                str(f.get("tempo", "—")),
                str(f.get("energy_mean", "—")),
                str(f.get("spectral_centroid_mean", "—")),
                str(f.get("zcr_mean", "—")),
                f"{round(f.get('silence_ratio', 0)*100, 1)}%",
                str(f.get("duration_sec", "—")),
            ])
    if len(feature_data) > 1:
        f1=2.4*cm; f2=1.8*cm; f3=2.0*cm; f4=2.8*cm; f5=2.6*cm; f6=2.0*cm
        f7=PAGE_W-f1-f2-f3-f4-f5-f6
        feat_table = Table(feature_data, colWidths=[f1, f2, f3, f4, f5, f6, f7])
        feat_table.setStyle(TableStyle(_base_table_style(extra=[
            ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ])))
        story.append(feat_table)

    # ── 4. Key Insights ───────────────────────────────────────
    story.append(Paragraph("4. Key Insights", section_style))
    insights = generate_insights(detected, all_features, all_predictions)
    for insight in insights:
        story.append(Paragraph(f"• {insight}", body_style))

    # ── Footer ────────────────────────────────────────────────
    story.append(Spacer(1, 0.4*cm))
    story.append(HRFlowable(width="100%", thickness=0.4, color=colors.HexColor("#dddddd"), spaceAfter=4))
    story.append(Paragraph(
        "AI Stem Separator  ·  Meta Demucs  ·  librosa  ·  GTZAN Dataset",
        small_style
    ))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()

    if output_path:
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)

    return pdf_bytes


def generate_insights(detected, all_features, all_predictions):
    insights = []
    if not detected:
        return ["No stems were detected in this track."]

    energies = {s: all_features[s]["energy_mean"] for s in detected if all_features[s]}
    if energies:
        dominant = max(energies, key=energies.get)
        insights.append(
            f"Most energetically dominant stem: '{dominant}' (mean energy {energies[dominant]})."
        )

    tempos = {s: all_features[s]["tempo"] for s in detected if all_features[s]}
    if len(tempos) > 1:
        vals = list(tempos.values())
        if max(vals) - min(vals) < 5:
            insights.append(f"All stems agree on a consistent tempo of ~{round(sum(vals)/len(vals))} BPM.")
        else:
            insights.append(f"Tempo varies across stems ({min(vals)}-{max(vals)} BPM) — complex rhythmic layering.")

    genres = [all_predictions[s]["top_genre"] for s in detected if all_predictions.get(s)]
    if genres:
        top = max(set(genres), key=genres.count)
        insights.append(f"{genres.count(top)} of {len(genres)} stems classified as '{top}'.")

    silent = [s for s in detected if all_features[s] and all_features[s].get("silence_ratio", 0) > 0.3]
    if silent:
        insights.append(f"High silence ratio (>30%) in: {', '.join(silent)} — sparse instruments.")

    bright = {s: all_features[s]["spectral_centroid_mean"] for s in detected if all_features[s]}
    if bright:
        b = max(bright, key=bright.get)
        insights.append(f"Brightest tonal character: '{b}' (spectral centroid {bright[b]} Hz).")

    return insights


def generate_csv(song_name, all_features, all_predictions):
    rows = []
    for stem, features in all_features.items():
        if features is None:
            continue
        row = {"song": song_name, "stem": stem}
        row.update(features)
        pred = all_predictions.get(stem)
        if pred:
            row["predicted_genre"]      = pred.get("top_genre", "")
            row["genre_confidence_pct"] = pred.get("confidence", "")
        rows.append(row)
    if not rows:
        return b""
    df = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()