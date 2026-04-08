# test_features.py
from features import extract_all_stems

# Load your last separated stems from the outputs folder to test
import os

stem_data = {}
stems_dir = None

for root, dirs, files in os.walk("outputs"):
    if root == "outputs":
        continue
    mp3s = [f for f in files if f.endswith(".mp3")]
    if mp3s:
        stems_dir = root
        break

if stems_dir:
    for stem in ["vocals", "drums", "bass", "guitar", "piano", "other"]:
        path = os.path.join(stems_dir, f"{stem}.mp3")
        if os.path.exists(path):
            with open(path, "rb") as f:
                stem_data[stem] = f.read()
        else:
            stem_data[stem] = None

    results = extract_all_stems(stem_data)
    for stem, feats in results.items():
        print(f"\n{stem}:")
        if feats:
            for k, v in feats.items():
                print(f"  {k}: {v}")
        else:
            print("  Not detected")
else:
    print("No stems found — separate a song first then run this")

from classifier import predict_all_stems

print("\n=== GENRE PREDICTIONS ===")
predictions = predict_all_stems(results)
for stem, pred in predictions.items():
    if pred:
        print(f"\n{stem}:")
        print(f"  Top genre: {pred['top_genre']} ({pred['confidence']}%)")
        print(f"  Top 3: {pred['top_3']}")
    else:
        print(f"\n{stem}: Not detected")

from reporter import generate_report, generate_csv

print("\n=== GENERATING REPORT ===")
pdf_bytes = generate_report("Someone Like You", results, predictions)
with open("test_report.pdf", "wb") as f:
    f.write(pdf_bytes)
print("PDF saved as test_report.pdf")

csv_bytes = generate_csv("Someone Like You", results, predictions)
with open("test_report.csv", "wb") as f:
    f.write(csv_bytes)
print("CSV saved as test_report.csv")