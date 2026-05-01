import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.predict import predict_sentiment


def analyze_sentiment(text):
    model_path = os.path.join(ROOT_DIR, 'model', 'model.pkl')
    vectorizer_path = os.path.join(ROOT_DIR, 'model', 'vectorizer.pkl')
    prediction, confidence = predict_sentiment(
        text,
        model_path=model_path,
        vectorizer_path=vectorizer_path,
    )
    return prediction, confidence
