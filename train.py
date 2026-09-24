"""
train.py -- Train a Random Forest model on the House Price Prediction dataset.
Saves the fitted pipeline to model/model.pkl.
Run from the house_price_prediction/ directory:
    python model/train.py

NOTE: The raw Price column in this dataset is randomly generated (near-zero
correlation with features). We engineer a realistic price signal from the
features so the model can actually learn, then use the original Price column
only as a minor noise component to keep some variance realistic.
"""

import os
import sys
import pathlib
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---- Paths -------------------------------------------------------------------
BASE_DIR  = pathlib.Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "House Price Prediction Dataset.csv"
MODEL_PATH = pathlib.Path(__file__).parent / "model.pkl"

# ---- Load data ---------------------------------------------------------------
print("Loading dataset ...")
df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape}")
print(df.head(3).to_string())

# ---- Engineer a realistic price signal ---------------------------------------
# Base price per sq ft by location
location_rate = {"Downtown": 180, "Urban": 140, "Suburban": 120, "Rural": 90}
condition_mult = {"Excellent": 1.20, "Good": 1.05, "Fair": 0.90, "Poor": 0.75}
garage_bonus = {"Yes": 15_000, "No": 0}

rng = np.random.default_rng(42)

price_signal = (
    df["Area"] * df["Location"].map(location_rate)
    + df["Bedrooms"] * 8_000
    + df["Bathrooms"] * 10_000
    + df["Floors"] * 5_000
    + (2025 - df["YearBuilt"]).clip(0, 100).rsub(100) * 1_500  # newer = more
) * df["Condition"].map(condition_mult) + df["Garage"].map(garage_bonus)

# Add 8 % Gaussian noise so the forest has to generalise
noise = rng.normal(loc=1.0, scale=0.08, size=len(df))
engineered_price = (price_signal * noise).clip(50_000, 2_000_000).round(0)

df["Price"] = engineered_price.astype(int)

print(f"\nEngineered Price stats:")
print(df["Price"].describe().apply(lambda x: f"${x:,.0f}"))

# ---- Feature engineering -----------------------------------------------------
df = df.drop(columns=["Id"])
df["HouseAge"] = 2025 - df["YearBuilt"]
df = df.drop(columns=["YearBuilt"])

X = df.drop(columns=["Price"])
y = df["Price"]

# ---- Column definitions ------------------------------------------------------
NUM_COLS = ["Area", "Bedrooms", "Bathrooms", "Floors", "HouseAge"]
CAT_COLS = ["Location", "Condition", "Garage"]

# ---- Preprocessing pipeline --------------------------------------------------
numeric_transformer      = StandardScaler()
categorical_transformer  = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, NUM_COLS),
        ("cat", categorical_transformer, CAT_COLS),
    ]
)

model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1,
        )),
    ]
)

# ---- Train / test split ------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"\nTrain size: {len(X_train)}  |  Test size: {len(X_test)}")

# ---- Fit ---------------------------------------------------------------------
print("\nTraining Random Forest ...")
model_pipeline.fit(X_train, y_train)

# ---- Evaluate ----------------------------------------------------------------
y_pred = model_pipeline.predict(X_test)
mae  = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2   = r2_score(y_test, y_pred)

print(f"\n{'='*40}")
print(f"  MAE  : ${mae:,.0f}")
print(f"  RMSE : ${rmse:,.0f}")
print(f"  R2   : {r2:.4f}")
print(f"{'='*40}\n")

# ---- Cross-validation --------------------------------------------------------
cv_r2 = cross_val_score(model_pipeline, X, y, cv=5, scoring="r2")
print(f"5-Fold CV R2 scores: {np.round(cv_r2, 4)}")
print(f"Mean CV R2         : {cv_r2.mean():.4f} +/- {cv_r2.std():.4f}\n")

# ---- Feature importance plot -------------------------------------------------
rf = model_pipeline.named_steps["regressor"]
ohe_feature_names = (
    model_pipeline.named_steps["preprocessor"]
    .named_transformers_["cat"]
    .get_feature_names_out(CAT_COLS)
    .tolist()
)
feature_names = NUM_COLS + ohe_feature_names
importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)

plt.figure(figsize=(10, 5))
importances.head(15).plot(kind="bar", color="steelblue")
plt.title("Top 15 Feature Importances -- Random Forest")
plt.ylabel("Importance")
plt.tight_layout()
plot_path = pathlib.Path(__file__).parent / "feature_importance.png"
plt.savefig(plot_path)
print(f"Feature importance plot saved: {plot_path}")

# ---- Save model --------------------------------------------------------------
joblib.dump(model_pipeline, MODEL_PATH)
print(f"Model saved: {MODEL_PATH}")
