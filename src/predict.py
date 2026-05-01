import os
import sys
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.preprocess import load_pickle, preprocess_text


def predict_sentiment(text, model_path=None, vectorizer_path=None, confidence_threshold=0.65):
    """Predict sentiment with confidence threshold logic.
    
    Args:
        text: Input text to analyze
        model_path: Path to saved model
        vectorizer_path: Path to saved vectorizer
        confidence_threshold: Return 'Uncertain' if confidence is below this (default: 0.65)
    
    Returns:
        Tuple of (prediction, confidence)
    """
    if model_path is None:
        model_path = os.path.join(ROOT_DIR, 'model', 'model.pkl')
    if vectorizer_path is None:
        vectorizer_path = os.path.join(ROOT_DIR, 'model', 'vectorizer.pkl')

    model = load_pickle(model_path)
    vectorizer = load_pickle(vectorizer_path)
    cleaned_text = preprocess_text(text)
    X = vectorizer.transform([cleaned_text])

    prediction = model.predict(X)[0]
    confidence = 1.0

    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)[0]
        predicted_index = int(np.where(model.classes_ == prediction)[0][0])
        confidence = float(probabilities[predicted_index])
    elif hasattr(model, 'decision_function'):
        score = float(model.decision_function(X)[0])
        confidence = 1 / (1 + np.exp(-score))

    # Apply confidence threshold: if below threshold, mark as uncertain
    if confidence < confidence_threshold:
        normalized_prediction = 'Uncertain'
    else:
        normalized_prediction = str(prediction).capitalize()
    
    return normalized_prediction, confidence


def test_negation_handling():
    """Test cases to verify negation handling and model accuracy."""
    test_cases = [
        ('not good at all', 'Negative'),
        ('I am not happy', 'Negative'),
        ('never buying again', 'Negative'),
        ('this is amazing', 'Positive'),
        ('love this movie', 'Positive'),
        ('absolutely terrible', 'Negative'),
        ('not bad actually', 'Positive'),
    ]
    
    print('\n=== Negation Handling Test Cases ===')
    for text, expected in test_cases:
        prediction, confidence = predict_sentiment(text)
        status = 'OK' if prediction.lower() == expected.lower() else 'MISMATCH'
        print(f'[{status}] "{text}" -> {prediction} ({confidence:.2f}) [Expected: {expected}]')


if __name__ == '__main__':
    # Run test cases
    test_negation_handling()
    
    # Example prediction
    print('\n=== Example Prediction ===')
    sample = 'I really enjoy this application.'
    prediction, confidence = predict_sentiment(sample)
    print(f'Text: "{sample}"')
    print(f'Prediction: {prediction}')
    print(f'Confidence: {confidence:.2f}')
