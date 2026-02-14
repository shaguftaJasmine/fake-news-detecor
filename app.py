"""
FAKE NEWS DETECTOR - FLASK WEB APP
Run with: python app.py
Visit: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load models
model = joblib.load('fakenews_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news_text = request.form['news']
        
        # Transform and predict
        features = vectorizer.transform([news_text.lower()])
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0]
        
        result = {
            'prediction': 'REAL' if prediction == 1 else 'FAKE',
            'fake_probability': float(proba[0]),
            'real_probability': float(proba[1])
        }
        
        return jsonify(result)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """REST API endpoint"""
    data = request.get_json()
    news_text = data['text']
    
    features = vectorizer.transform([news_text.lower()])
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    
    return jsonify({
        'is_fake': bool(prediction == 0),
        'confidence': float(max(proba)),
        'fake_probability': float(proba[0]),
        'real_probability': float(proba[1])
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)