import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np


def evaluate_model():
    # 1. Load the model and the test data
    model = joblib.load("./models/model.pkl")
    # Assuming you saved your split data during the training phase
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv")

    # 2. Make Predictions
    predictions = model.predict(X_test)

    # 3. Calculate Regression Metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"Mean Absolute Error (MAE): Rs. {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): Rs. {rmse:.2f}")
    print(f"R-squared Score (R2): {r2:.4f}")

    # 4. Generate Visualization (Actual vs Predicted)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual vs. Predicted Electronics Prices (Ikman.lk)")
    plt.savefig("notebooks/evaluation_plot.png")
    plt.show()


if __name__ == "__main__":
    evaluate_model()
