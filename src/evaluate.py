import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_regression():
    # 1. Load the saved pipeline and test data
    if not os.path.exists("models/model.pkl"):
        print("Model not found. Please run train.py first.")
        return

    pipeline = joblib.load("models/model.pkl")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv")

    # 2. Generate Predictions
    y_pred = pipeline.predict(X_test)

    # 3. Calculate Regression Metrics
    # Flatten y_test to 1D array to avoid dimension issues
    y_test_values = y_test.values.flatten()

    mae = mean_absolute_error(y_test_values, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_values, y_pred))
    r2 = r2_score(y_test_values, y_pred)

    print("\n" + "=" * 30)
    print(" REGRESSION MODEL EVALUATION ")
    print("=" * 30)
    print(f"Mean Absolute Error (MAE):    Rs. {mae:,.2f}")
    print(f"Root Mean Squared Error (RMSE): Rs. {rmse:,.2f}")
    print(f"R-squared Score (R2):         {r2:.4f}")
    print("=" * 30)

    # 4. Visualization FIX
    plt.figure(figsize=(10, 6))
    plt.scatter(
        y_test_values, y_pred, alpha=0.5, color="royalblue", label="Predictions"
    )

    # Use numpy to find the absolute max/min for the reference line
    all_values = np.concatenate([y_test_values, y_pred])
    min_val = all_values.min()
    max_val = all_values.max()

    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Fit")

    plt.xlabel("Actual Price (Rs.)")
    plt.ylabel("Predicted Price (Rs.)")
    plt.title("Ikman.lk Price Prediction: Actual vs. Predicted")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    os.makedirs("reports/plots", exist_ok=True)
    plt.savefig("reports/plots/evaluation_scatter.png")
    print("\n✅ Evaluation plot saved to reports/plots/evaluation_scatter.png")
    plt.show()


if __name__ == "__main__":
    evaluate_regression()
