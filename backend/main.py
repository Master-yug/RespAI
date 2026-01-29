import os
import pickle
import json
import numpy as np

from flask import Flask, request, jsonify
from flask_cors import CORS

from utils.audio_features import extract_features

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODELS_FOLDER = os.path.join(BASE_DIR, "models")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---------------------------------------------------------------------
# GLOBALS FOR MODEL COMPONENTS
# ---------------------------------------------------------------------
model = None
scaler = None
label_encoder = None
feature_columns = None


def load_ai_model():
    """
    Load RandomForest model, scaler, label encoder and feature column order.
    """
    global model, scaler, label_encoder, feature_columns

    try:
        # model
        model_path = os.path.join(MODELS_FOLDER, "diagnosis_model.pkl")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print(f"✓ Model loaded from: {model_path}")

        # scaler
        scaler_path = os.path.join(MODELS_FOLDER, "scaler.pkl")
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        print(f"✓ Scaler loaded from: {scaler_path}")

        # label encoder
        encoder_path = os.path.join(MODELS_FOLDER, "label_encoder.pkl")
        with open(encoder_path, "rb") as f:
            label_encoder = pickle.load(f)
        print(f"✓ Label encoder loaded from: {encoder_path}")

        # feature columns
        columns_path = os.path.join(MODELS_FOLDER, "feature_columns.json")
        with open(columns_path, "r", encoding="utf-8") as f:
            feature_data = json.load(f)
        feature_columns = feature_data["FEATURE_COLUMNS"]
        print(f"✓ Feature columns loaded: {len(feature_columns)} features")

        print("Model system ready for predictions!")

    except FileNotFoundError as e:
        print(f"✗ Error: Model file not found: {e}")
        print("Running in simulation mode (no real predictions).")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("Running in simulation mode (no real predictions).")


load_ai_model()

# ---------------------------------------------------------------------
# CORE PREDICTION
# ---------------------------------------------------------------------
def predict_disease(file_path: str):
    """
    Extract features from audio file, run model, and return diagnosis.
    """
    print(f"Received file: {file_path}")

    if model is None or scaler is None or label_encoder is None or feature_columns is None:
        raise RuntimeError("Model not loaded correctly. Check model files in 'models/'.")

    # 1) Extract features from audio
    features_dict = extract_features(file_path)
    if not isinstance(features_dict, dict):
        raise ValueError("extract_features did not return a dict.")

    # 2) Build feature vector in the exact training order
    feature_vector = []
    for name in feature_columns:
        value = features_dict.get(name, 0.0)
        # replace NaN / inf with 0.0 to keep scaler and model happy
        if isinstance(value, (float, int)) and (np.isnan(value) or np.isinf(value)):
            value = 0.0
        feature_vector.append(float(value))

    X = np.array([feature_vector], dtype=float)

    print(f"Feature vector length: {X.shape[1]} (expected {len(feature_columns)})")

    # 3) Scale
    X_scaled = scaler.transform(X)

    # 4) Predict
    proba = model.predict_proba(X_scaled)[0]
    pred_idx = int(np.argmax(proba))
    diagnosis = label_encoder.inverse_transform([pred_idx])[0]
    confidence = float(np.max(proba))

    probabilities = {
        str(label): float(p) for label, p in zip(label_encoder.classes_, proba)
    }

    print(f"Predicted: {diagnosis} (confidence: {confidence:.2%})")

    return {
        "diagnosis": str(diagnosis),
        "confidence": confidence,
        "probabilities": probabilities,
    }


# ---------------------------------------------------------------------
# FLASK ROUTES
# ---------------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    filename = file.filename
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    try:
        result = predict_disease(file_path)
        # optional: os.remove(file_path)
        return jsonify(result)
    except Exception as e:
        import traceback

        print("ERROR IN /predict:")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Starting server...")
    app.run(debug=True, port=5000)
