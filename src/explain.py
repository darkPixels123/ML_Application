import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os
import numpy as np


def explain_model():
    # 1. Load the saved pipeline and test data
    pipeline = joblib.load("models/model.pkl")
    X_test = pd.read_csv("data/processed/X_test.csv")

    # 2. Extract components
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    # 3. Transform and FIX format
    # Convert sparse matrix to dense array so SHAP can read it
    X_test_transformed = preprocessor.transform(X_test)

    if hasattr(X_test_transformed, "toarray"):
        X_test_transformed = X_test_transformed.toarray()

    # Ensure it's float64, not Object 'O'
    X_test_transformed = X_test_transformed.astype(np.float64)

    # Get feature names for the plot
    feature_names = preprocessor.get_feature_names_out()

    # 4. Initialize SHAP
    explainer = shap.TreeExplainer(model)

    # 5. Calculate SHAP values
    # We use X_test_transformed[:100] if your dataset is huge to save time
    shap_values = explainer.shap_values(X_test_transformed)

    # 6. Plotting
    plt.figure(figsize=(12, 8))

    # For Classifiers, shap_values is a list. Index [1] is the "Overpriced" class.
    # If shap_values is just an array, use it directly.
    if isinstance(shap_values, list):
        display_values = shap_values[1]
    else:
        display_values = shap_values

    shap.summary_plot(
        display_values, X_test_transformed, feature_names=feature_names, show=False
    )

    os.makedirs("notebooks", exist_ok=True)
    plt.savefig("notebooks/shap_summary_plot.png", bbox_inches="tight")
    print("✅ SHAP Summary Plot saved to notebooks/shap_summary_plot.png")
    plt.show()


if __name__ == "__main__":
    explain_model()
