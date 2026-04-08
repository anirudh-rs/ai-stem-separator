# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

print("Loading GTZAN dataset...")
df = pd.read_csv("features_30_sec.csv")

# ── Drop non-feature columns ──────────────────────────────────
df = df.drop(columns=["filename", "length"], errors="ignore")

# ── Separate features and labels ─────────────────────────────
X = df.drop(columns=["label"])
y = df["label"]

# ── Encode genre labels to numbers ───────────────────────────
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"Genres found: {list(le.classes_)}")
print(f"Total samples: {len(X)}")

# ── Train/test split ──────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# ── Train Random Forest ───────────────────────────────────────
print("\nTraining Random Forest classifier...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.1f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ── Save model and encoder ────────────────────────────────────
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/genre_model.pkl")
joblib.dump(le, "model/label_encoder.pkl")

# ── Save feature column names ─────────────────────────────────
# We need these to align our stem features with the model's expected input
joblib.dump(list(X.columns), "model/feature_columns.pkl")

print("\nModel saved to model/genre_model.pkl")
print("Label encoder saved to model/label_encoder.pkl")
print("Feature columns saved to model/feature_columns.pkl")