"""
train_model.py
==============
Run this script once to train the Gradient Boosting model (best after
hyperparameter tuning, CV R² = 0.8963, Test R² = 0.9429) and save:

    models/gradient_boosting_model.pkl   ← the trained regressor
    models/preprocessor.pkl             ← fitted ColumnTransformer
    models/label_encoder_model.pkl      ← fitted LabelEncoder for 'model' col

Usage
-----
    python train_model.py --data path/to/cardekho_imputated.csv

The CSV must be the same cardekho_imputated.csv used in your notebooks
(index column present, 13 features + target).
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# ── Exact best parameters from hyperparameter tuning ─────────────────
BEST_GB_PARAMS = dict(
    subsample=1.0,
    n_estimators=300,
    min_samples_split=2,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
)

ONEHOT_COLS  = ["seller_type", "fuel_type", "transmission_type"]
LABEL_COL    = "model"
TARGET_COL   = "selling_price"
DROP_COLS    = ["car_name", "brand"]


def build_preprocessor(num_features):
    return ColumnTransformer(
        transformers=[
            ("OneHotEncoder", OneHotEncoder(drop="first"), ONEHOT_COLS),
            ("StandardScaler", StandardScaler(),            list(num_features)),
        ],
        remainder="passthrough",
    )


def main(data_path: str):
    os.makedirs("models", exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────
    df = pd.read_csv(data_path, index_col=[0])
    print(f"Loaded dataset: {df.shape}")

    # ── Clean ─────────────────────────────────────────────────────────
    for col in DROP_COLS:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    X = df.drop([TARGET_COL], axis=1)
    y = df[TARGET_COL]

    # ── LabelEncode 'model' ───────────────────────────────────────────
    le = LabelEncoder()
    X[LABEL_COL] = le.fit_transform(X[LABEL_COL])

    # ── Build preprocessor ────────────────────────────────────────────
    num_features = X.select_dtypes(exclude="object").columns
    preprocessor = build_preprocessor(num_features)

    X_transformed = preprocessor.fit_transform(X)

    # ── Train / test split ────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, test_size=0.2, random_state=42
    )

    # ── Train best model ──────────────────────────────────────────────
    print("\nTraining Gradient Boosting Regressor (best params)…")
    gb = GradientBoostingRegressor(**BEST_GB_PARAMS)
    gb.fit(X_train, y_train)

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_pred = gb.predict(X_test)
    print(f"  Test  R²   : {r2_score(y_test, y_pred):.4f}")
    print(f"  Test  MAE  : {mean_absolute_error(y_test, y_pred):,.0f}")
    print(f"  Test  RMSE : {np.sqrt(mean_squared_error(y_test, y_pred)):,.0f}")

    # ── Save artefacts ────────────────────────────────────────────────
    pickle.dump(gb,           open("models/gradient_boosting_model.pkl", "wb"))
    pickle.dump(preprocessor, open("models/preprocessor.pkl",            "wb"))
    pickle.dump(le,           open("models/label_encoder_model.pkl",     "wb"))

    # Save model unique values for the UI dropdowns
    model_classes = list(le.classes_)
    import json
    with open("models/model_classes.json", "w") as f:
        json.dump(model_classes, f)

    print("\nSaved:")
    print("  models/gradient_boosting_model.pkl")
    print("  models/preprocessor.pkl")
    print("  models/label_encoder_model.pkl")
    print("  models/model_classes.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="datasets/cardekho_imputated.csv",
        help="Path to cardekho_imputated.csv",
    )
    args = parser.parse_args()
    main(args.data)
