ode# Bug Fix Plan

## Problem
- `TypeError: NDFrame.fillna() got an unexpected keyword argument 'method'` in `preprocess.py` line 53.
- `app.py` does not save `scaler.pkl`, `label_encoder.pkl`, or `feature_names.pkl` after training, causing Prediction page to load `None` artifacts.\n

1. [x] Fix `preprocess.py`: Replace deprecated `.fillna(method="bfill")` with `.bfill()`.
2. [x] Fix `app.py`: Import and call `save_artifacts()` after training to persist scaler, label encoder, and feature names.
3. [x] Run Streamlit app and verify Model Training page works without errors.
