from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("gacha_model.keras")

MAX_ROLLS = {1: 90, 2: 80}

app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    roll = int(data.get("roll", 1))
    banner_type = int(data.get("banner_type", 1))

    if banner_type not in MAX_ROLLS:
        return jsonify({"error": "Invalid banner_type"}), 400
    if not (1 <= roll <= MAX_ROLLS[banner_type]):
        return jsonify({"error": f"Roll must be 1–{MAX_ROLLS[banner_type]}"}), 400

    max_roll = MAX_ROLLS[banner_type]

    # Predict local probabilities p_i for i = 1..roll
    rolls = np.arange(1, roll + 1)
    X_roll = (rolls / max_roll).reshape(-1, 1).astype("float32")
    X_banner = np.full((roll,), banner_type, dtype="int32").reshape(-1, 1)

    local_probs = model.predict([X_roll, X_banner], verbose=0).flatten()

    # Cumulative probability
    cumulative_prob = 1 - np.prod(1 - local_probs)

    # Hard pity
    if (banner_type == 1 and roll == 90) or (banner_type == 2 and roll == 80):
        cumulative_prob = 1.0

    prob_percent = f"{cumulative_prob * 100:.2f}%"

    return jsonify({
        "probability_percent": prob_percent
    })
