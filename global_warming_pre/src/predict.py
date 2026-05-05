"""
src/predict.py
Prediction utilities for the Climate AI Dashboard.
"""

import os
import pickle
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
_SCALER_PATH   = os.path.join(MODELS_DIR, "scaler.pkl")
_LE_PATH       = os.path.join(MODELS_DIR, "label_encoder.pkl")
_FEATURES_PATH = os.path.join(MODELS_DIR, "feature_names.pkl")
_MODEL_PATH    = os.path.join(MODELS_DIR, "best_model.pkl")


def save_artifacts(scaler, le, feature_names):
    """Persist scaler, label encoder, and feature names alongside the model."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(_SCALER_PATH,   "wb") as f: pickle.dump(scaler,        f)
    with open(_LE_PATH,       "wb") as f: pickle.dump(le,            f)
    with open(_FEATURES_PATH, "wb") as f: pickle.dump(feature_names, f)


def load_best_model():
    """
    Load saved model + artifacts.

    Returns (model, scaler, le, feature_names) — all None if not found.
    """
    # Try session state first (populated right after training in app.py)
    try:
        import streamlit as st
        if "model_trained" in st.session_state:
            # Re-train is needed; artifacts live in session
            pass
    except Exception:
        pass

    if not os.path.exists(_MODEL_PATH):
        return None, None, None, None

    with open(_MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    scaler = pickle.load(open(_SCALER_PATH, "rb"))   if os.path.exists(_SCALER_PATH)   else None
    le     = pickle.load(open(_LE_PATH,     "rb"))   if os.path.exists(_LE_PATH)       else None
    feats  = pickle.load(open(_FEATURES_PATH,"rb"))  if os.path.exists(_FEATURES_PATH) else None

    return model, scaler, le, feats


def _encode_country(le, country: str) -> int:
    """Safely encode a country; return 0 if unseen."""
    if le is None:
        return 0
    try:
        return int(le.transform([country])[0])
    except Exception:
        return 0


def _build_feature_vector(feature_names, year, country_enc, co2, prev_temp):
    """Construct input vector matching training feature order."""
    mapping = {
        "Year":           year,
        "Year_sq":        year ** 2,
        "Year_norm":      (year - 1961) / 61,
        "Country_enc":    country_enc,
        "CO2":            co2,
        "TempChange":     0.0,           # unknown at prediction time
        "AvgTemp_lag1":   prev_temp,
        "AvgTemp_lag3":   prev_temp,
        "AvgTemp_roll5":  prev_temp,
    }
    if feature_names is None:
        feature_names = list(mapping.keys())
    return np.array([[mapping.get(f, 0.0) for f in feature_names]])


def predict_temperature(model, scaler, le, feature_names,
                        year: int, country: str, co2: float, prev_temp: float):
    """
    Predict average temperature.

    Returns (predicted_temp: float, confidence: float [60-99])
    """
    country_enc = _encode_country(le, country)
    X = _build_feature_vector(feature_names, year, country_enc, co2, prev_temp)

    if scaler is not None:
        X = scaler.transform(X)

    pred = float(model.predict(X)[0])

    # Heuristic confidence score based on year distance from training data
    year_dist = max(0, year - 2022)
    confidence = max(60.0, 97.0 - year_dist * 1.2)

    return round(pred, 2), round(confidence, 1)
