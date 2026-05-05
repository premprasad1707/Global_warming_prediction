"""
src/preprocess.py
Data preprocessing pipeline for the Climate AI Dashboard.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def preprocess_data(df: pd.DataFrame):
    """
    Full preprocessing pipeline.

    Returns
    -------
    X_train, X_test, y_train, y_test, feature_names, scaler, le
    """
    df = df.copy()

    # ── 1. Ensure required columns ─────────────────────────────────────────
    required = {"Year", "AvgTemp"}
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataset. "
                             f"Available columns: {list(df.columns)}")

    # ── 2. Handle missing values ────────────────────────────────────────────
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in num_cols:
        df[col] = df[col].interpolate(method="linear").fillna(df[col].mean())

    # ── 3. Label-encode Country ─────────────────────────────────────────────
    le = LabelEncoder()
    if "Country" in df.columns:
        df["Country_enc"] = le.fit_transform(df["Country"].astype(str))
    else:
        df["Country_enc"] = 0
        le.fit(["Unknown"])

    # ── 4. Time-series feature engineering ─────────────────────────────────
    df = df.sort_values(["Country_enc", "Year"]).reset_index(drop=True)

    df["Year_sq"]   = df["Year"] ** 2            # non-linear year
    df["Year_norm"] = (df["Year"] - 1961) / 61   # normalized year in [0,1]

    # Lag features (previous 1 and 3 years) per country
    for lag in [1, 3]:
        df[f"AvgTemp_lag{lag}"] = (
            df.groupby("Country_enc")["AvgTemp"]
            .shift(lag)
            .bfill()
        )

    # 5-year rolling mean
    df["AvgTemp_roll5"] = (
        df.groupby("Country_enc")["AvgTemp"]
        .transform(lambda x: x.rolling(5, min_periods=1).mean())
    )

    # CO2 feature (if present)
    if "CO2" not in df.columns:
        df["CO2"] = 280 + (df["Year"] - 1961) * 1.8   # synthetic proxy

    # TempChange (if present)
    if "TempChange" not in df.columns:
        df["TempChange"] = df.groupby("Country_enc")["AvgTemp"].diff().fillna(0)

    # ── 5. Select features ──────────────────────────────────────────────────
    feature_cols = [
        "Year", "Year_sq", "Year_norm",
        "Country_enc",
        "CO2",
        "TempChange",
        "AvgTemp_lag1", "AvgTemp_lag3",
        "AvgTemp_roll5",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].values
    y = df["AvgTemp"].values

    # ── 6. Train / test split ───────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # ── 7. Feature scaling ──────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, feature_cols, scaler, le
