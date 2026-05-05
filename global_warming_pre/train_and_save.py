"""
train_and_save.py
Stand-alone script to train all models and save artifacts to /models/.
Run:  python train_and_save.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from src.preprocess import preprocess_data
from src.train      import train_models
from src.predict    import save_artifacts

CSV_PATH = os.path.join("data", "dataset.csv")

def synthetic_data():
    import numpy as np
    np.random.seed(42)
    years     = range(1961, 2023)
    countries = ["United States","China","India","Germany","Brazil",
                 "Australia","Russia","Canada","Japan","France"]
    rows = []
    base = {c: float(i*2 + 8) for i, c in enumerate(countries)}
    for c in countries:
        for i, y in enumerate(years):
            rows.append({
                "Year": y, "Country": c,
                "AvgTemp":    round(base[c] + 0.018*i + np.random.normal(0, .3), 3),
                "TempChange": round(0.018 + np.random.normal(0, .1), 3),
                "CO2":        round(250 + 1.8*i + np.random.normal(0, 8), 2),
            })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
    else:
        print("No dataset.csv found — using synthetic data.")
        df = synthetic_data()

    X_tr, X_te, y_tr, y_te, feat, scaler, le = preprocess_data(df)
    results, best_name, best_model, _ = train_models(X_tr, X_te, y_tr, y_te, feat)
    save_artifacts(scaler, le, feat)

    print("\n=== Results ===")
    for name, m in results.items():
        print(f"  {name}: RMSE={m['RMSE']}  MAE={m['MAE']}  R²={m['R2']}")
    print(f"\nBest: {best_name}")
    print("Artifacts saved to /models/")
