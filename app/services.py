import os
import sys
import joblib

# ✅ Fix path for production
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# ✅ Safe import AFTER path fix
from src.predict import predict_sentiment


# ✅ Debug logs (VERY IMPORTANT for Render)
print("📁 ROOT_DIR:", ROOT_DIR)
print("📂 Files in ROOT:", os.listdir(ROOT_DIR))

model_dir = os.path.join(ROOT_DIR, "model")
print("📂 Model folder exists:", os.path.exists(model_dir))
if os.path.exists(model_dir):
    print("📂 Model files:", os.listdir(model_dir))


def analyze_sentiment(text):
    try:
        model_path = os.path.join(ROOT_DIR, "model", "model.pkl")
        vectorizer_path = os.path.join(ROOT_DIR, "model", "vectorizer.pkl")

        print("🔍 Loading model from:", model_path)

        prediction, confidence = predict_sentiment(
            text,
            model_path=model_path,
            vectorizer_path=vectorizer_path,
        )

        return prediction, confidence

    except Exception as e:
        print("❌ ERROR in analyze_sentiment:", str(e))
        return "Error", 0.0