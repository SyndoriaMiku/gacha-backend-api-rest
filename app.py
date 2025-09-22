from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tflite_runtime.interpreter as tflite


# Load TFLite model and allocate tensors
interpreter = tflite.Interpreter(model_path="gacha_model.tflite")
interpreter.allocate_tensors()

# Input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Max rolls cho từng banner
MAX_ROLLS = {1: 90, 2: 80}

# Flask app
app = Flask(__name__)
CORS(app)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    roll = int(data.get("roll", 1))
    banner_type = int(data.get("banner_type", 1))

    # Validate input
    if banner_type not in MAX_ROLLS:
        return jsonify({"error": "Invalid banner_type"}), 400
    if not (1 <= roll <= MAX_ROLLS[banner_type]):
        return jsonify({"error": f"Roll must be 1–{MAX_ROLLS[banner_type]}"}), 400

    max_roll = MAX_ROLLS[banner_type]

    local_probs = []
    for r in range(1, roll + 1):
        # Chuẩn hóa roll
        roll_val = np.array([[r / max_roll]], dtype="float32")
        banner_val = np.array([[banner_type]], dtype="int32")

        # Set tensor cho từng input
        interpreter.set_tensor(input_details[0]['index'], roll_val)
        interpreter.set_tensor(input_details[1]['index'], banner_val)

        # Run model
        interpreter.invoke()

        # Lấy kết quả
        prob = interpreter.get_tensor(output_details[0]['index'])[0][0]
        local_probs.append(prob)

    local_probs = np.array(local_probs)

    # Cumulative probability
    cumulative_prob = 1 - np.prod(1 - local_probs)

    # Hard pity
    if (banner_type == 1 and roll == 90) or (banner_type == 2 and roll == 80):
        cumulative_prob = 1.0

    return jsonify({
        "probability_percent": f"{cumulative_prob * 100:.2f}%"
    })

