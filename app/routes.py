from flask import jsonify, render_template, request
from app.services import analyze_sentiment


def register_routes(app):
    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/predict', methods=['POST'])
    def predict():
        if not request.is_json:
            return jsonify({'error': 'Invalid JSON payload'}), 400

        data = request.get_json()
        if not isinstance(data, dict):
            return jsonify({'error': 'Invalid payload format'}), 400

        text = data.get('text', '')
        if not isinstance(text, str) or not text.strip():
            return jsonify({'error': 'Text is required'}), 400

        try:
            prediction, confidence = analyze_sentiment(text.strip())
            return jsonify({
                'prediction': prediction,
                'confidence': round(confidence, 2),
            })
        except Exception as exc:
            app.logger.exception('Prediction failed')
            return jsonify({'error': 'Unable to analyze sentiment at this time.'}), 500
