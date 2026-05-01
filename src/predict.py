import os
import sys
import numpy as np
import joblib

# ✅ Fix root path for deployment
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.preprocess import preprocess_text


def predict_sentiment(text, model_path=None, vectorizer_path=None):
    """
    Predict sentiment (Production-safe version)

    Args:
        text: input text
        model_path: optional custom model path
        vectorizer_path: optional custom vectorizer path

    Returns:
        (prediction, confidence)
    """

    # ✅ Default paths (Render-safe)
    if model_path is None:
        model_path = os.path.join(ROOT_DIR, "model", "model.pkl")

    if vectorizer_path is None:
        vectorizer_path = os.path.join(ROOT_DIR, "model", "vectorizer.pkl")

    try:
        # 🔥 Load model + vectorizer
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)

        # 🔥 Preprocess
        cleaned_text = preprocess_text(text)
        X = vectorizer.transform([cleaned_text])

        # 🔥 Prediction
        prediction = model.predict(X)[0]

        # 🔥 Confidence calculation
        confidence = 1.0

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
            pred_index = int(np.where(model.classes_ == prediction)[0][0])
            confidence = float(probs[pred_index])

        elif hasattr(model, "decision_function"):
            score = float(model.decision_function(X)[0])
            confidence = float(1 / (1 + np.exp(-score)))  # sigmoid

        # ✅ ALWAYS return valid label (NO "Uncertain")
        return str(prediction).capitalize(), confidence

    except Exception as e:
        print("❌ Prediction error:", str(e))
        return "Error", 0.0


# -----------------------------
# OPTIONAL LOCAL TEST
# -----------------------------
if __name__ == "__main__":
    sample = "I am not happy with this product"
    pred, conf = predict_sentiment(sample)
    print(f"Text: {sample}")
    print(f"Prediction: {pred}")
    print(f"Confidence: {conf:.2f}")