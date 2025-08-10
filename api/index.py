from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = joblib.load(__file__.replace("index.py", "heart_disease_model.pkl"))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([list(data.values())]).astype(float)
    prediction = model.predict(features)[0]
    return jsonify({'prediction': int(prediction)})
