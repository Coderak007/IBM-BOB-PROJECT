"""
app.py — Flask REST API for House Price Prediction.
Endpoints:
  GET  /health        → service health check
  POST /predict       → predict price from house features
  GET  /model-info    → returns model metadata & feature ranges
"""

import pathlib
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ── Load model once at startup ────────────────────────────────────────────────
MODEL_PATH = pathlib.Path(__file__).parent / "model" / "model.pkl"
pipeline = joblib.load(MODEL_PATH)
print(f"[INFO] Model loaded from {MODEL_PATH}")

FEATURE_COLS   = ["Area", "Bedrooms", "Bathrooms", "Floors", "HouseAge",
                  "Location", "Condition", "Garage"]
LOCATIONS      = ["Downtown", "Suburban", "Urban", "Rural"]
CONDITIONS     = ["Excellent", "Good", "Fair", "Poor"]
GARAGE_OPTIONS = ["Yes", "No"]

# ── Helpers ───────────────────────────────────────────────────────────────────
def build_dataframe(data: dict) -> pd.DataFrame:
    year_built = int(data.get("YearBuilt", 2000))
    house_age  = 2025 - year_built

    row = {
        "Area":      float(data["Area"]),
        "Bedrooms":  int(data["Bedrooms"]),
        "Bathrooms": int(data["Bathrooms"]),
        "Floors":    int(data["Floors"]),
        "HouseAge":  house_age,
        "Location":  str(data["Location"]),
        "Condition": str(data["Condition"]),
        "Garage":    str(data["Garage"]),
    }
    return pd.DataFrame([row])

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "RandomForestRegressor"})


@app.route("/model-info", methods=["GET"])
def model_info():
    return jsonify({
        "features": FEATURE_COLS,
        "locations": LOCATIONS,
        "conditions": CONDITIONS,
        "garage_options": GARAGE_OPTIONS,
        "year_range": {"min": 1900, "max": 2024},
        "area_range": {"min": 300, "max": 10000, "unit": "sq ft"},
    })


@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json(force=True, silent=True)
    if not body:
        return jsonify({"error": "Request body must be JSON"}), 400

    required = ["Area", "Bedrooms", "Bathrooms", "Floors", "YearBuilt",
                "Location", "Condition", "Garage"]
    missing = [f for f in required if f not in body]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    # Validate categoricals
    if body["Location"] not in LOCATIONS:
        return jsonify({"error": f"Location must be one of {LOCATIONS}"}), 400
    if body["Condition"] not in CONDITIONS:
        return jsonify({"error": f"Condition must be one of {CONDITIONS}"}), 400
    if body["Garage"] not in GARAGE_OPTIONS:
        return jsonify({"error": f"Garage must be one of {GARAGE_OPTIONS}"}), 400

    try:
        df = build_dataframe(body)
        price = float(pipeline.predict(df)[0])

        # Confidence band ±10 % using tree variance
        estimators = pipeline.named_steps["regressor"].estimators_
        preprocessed = pipeline.named_steps["preprocessor"].transform(df)
        tree_preds = np.array([t.predict(preprocessed)[0] for t in estimators])
        ci_low  = float(np.percentile(tree_preds, 5))
        ci_high = float(np.percentile(tree_preds, 95))

        return jsonify({
            "predicted_price": round(price, 2),
            "confidence_interval": {
                "low":  round(ci_low, 2),
                "high": round(ci_high, 2),
            },
            "currency": "USD",
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
