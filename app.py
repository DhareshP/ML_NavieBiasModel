from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the pre-trained model and vectorizer from .pkl files
try:
    model = joblib.load('naive_bayes_model.pkl')  # Ensure you provide the correct path
    vectorizer = joblib.load('vectorizer.pkl')  # Ensure you provide the correct path
except FileNotFoundError as e:
    print(f"Error loading model: {e}")
    model, vectorizer = None, None  # Ensure app doesn't crash if model loading fails


@app.route('/analyze', methods=['POST'])
def analyze():
    if model is None or vectorizer is None:
        return jsonify({'error': 'Model or vectorizer not found'}), 500

    data = request.json
    comments = data.get('comments', [])

    if not comments:
        return jsonify({'error': 'No comments provided'}), 400

    comments_vectorized = vectorizer.transform(comments)
    predictions_proba = model.predict_proba(comments_vectorized)

    results = [{'comment': comment, 'sentiment_score': prob[1],
                'sentiment_label': 'positive' if prob[1] >= 0.5 else 'negative'}
               for comment, prob in zip(comments, predictions_proba)]

    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True)
