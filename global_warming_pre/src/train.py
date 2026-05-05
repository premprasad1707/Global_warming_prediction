"""
src/train.py
Model training pipeline for the Climate AI Dashboard.
Trains Linear Regression, Random Forest, XGBoost, and LightGBM.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def _evaluate(model, X_test, y_test):
    """Compute RMSE, MAE, and R² for a fitted model."""
    preds = model.predict(X_test)
    rmse  = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae   = float(mean_absolute_error(y_test, preds))
    r2    = float(r2_score(y_test, preds))
    return {"RMSE": round(rmse, 4), "MAE": round(mae, 4), "R2": round(r2, 4)}


def _feature_importance(model, feature_names):
    """Extract feature importance if available, return sorted DataFrame."""
    fi = None
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
    elif hasattr(model, "coef_"):
        fi = np.abs(model.coef_)
    if fi is None:
        return None
    df_fi = pd.DataFrame({"Feature": feature_names, "Importance": fi})
    return df_fi.sort_values("Importance", ascending=False).reset_index(drop=True)


def train_models(X_train, X_test, y_train, y_test, feature_names):
    """
    Train all models, evaluate, save best model to /models/best_model.pkl.

    Returns
    -------
    results : dict  {model_name: {RMSE, MAE, R2}}
    best_name : str
    best_model : fitted sklearn model
    fi_df : DataFrame | None   feature importance of best model
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest":     RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            random_state=42, verbosity=0, n_jobs=-1,
        )
    if HAS_LGBM:
        models["LightGBM"] = LGBMRegressor(
            n_estimators=200, learning_rate=0.05, num_leaves=31,
            random_state=42, n_jobs=-1, verbose=-1,
        )

    results    = {}
    fitted     = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = _evaluate(model, X_test, y_test)
        results[name] = metrics
        fitted[name]  = model
        print(f"  {name}: RMSE={metrics['RMSE']}  MAE={metrics['MAE']}  R²={metrics['R2']}")

    # ── Select best by RMSE ─────────────────────────────────────────────────
    best_name  = min(results, key=lambda k: results[k]["RMSE"])
    best_model = fitted[best_name]

    # ── Save ────────────────────────────────────────────────────────────────
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"\nBest model '{best_name}' saved to {model_path}")

    fi_df = _feature_importance(best_model, feature_names)
    return results, best_name, best_model, fi_df
