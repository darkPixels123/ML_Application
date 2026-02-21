import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Import your custom modules
from src.data_processing import process_data
from src.feature_engineering import apply_feature_engineering


def train_regression_model():
    # 1. Load Raw Data
    # Now returns a DF containing 'Title' for engineering
    df = process_data("./data/raw/ikman_market_data.csv")

    if df is None:
        print("Error: Could not load data.")
        return

    # 2. Apply Feature Engineering
    # Title is processed here and then dropped inside this function
    df = apply_feature_engineering(df)

    # 3. Define Features and Target
    # ADDED: 'Has_Warranty' to match your engineering script
    features = [
        "Brand",
        "Location",
        "Is_Member",
        "Is_Promoted",
        "Brand_Avg_Price",
        "Is_Brand_New",
        "Has_Warranty",
    ]

    # Check if all features exist (prevents KeyError if engineering changes)
    features = [f for f in features if f in df.columns]

    X = df[features]
    y = df["Price"]

    # 4. Define Preprocessing Pipeline
    categorical_cols = ["Brand", "Location"]
    # We scale all continuous numerical features
    numerical_cols = ["Brand_Avg_Price"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numerical_cols),
        ],
        remainder="passthrough",
    )

    # 5. Define Regression Model
    model = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42)

    # 6. Create the Full Pipeline
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    # 7. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 8. Train
    print(f"🚀 Training Regression Model with {len(features)} features...")
    pipeline.fit(X_train, y_train)

    # 9. Evaluate
    y_pred = pipeline.predict(X_test)
    print(f"\n--- Performance ---")
    print(f"MAE: Rs. {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")

    # 10. Save for next steps
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    joblib.dump(pipeline, "models/model.pkl")
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    print("\n✅ Regression pipeline saved to models/model.pkl")


if __name__ == "__main__":
    train_regression_model()
