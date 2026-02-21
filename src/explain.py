import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os
import numpy as np


def explain_regression_model():
    # 1. Load components
    pipeline = joblib.load("models/model.pkl")
    X_test = pd.read_csv("data/processed/X_test.csv")

    # 2. Extract model and preprocessor
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    # 3. Transform data for SHAP
    # We must transform the test data so it's numeric for the TreeExplainer
    X_test_transformed = preprocessor.transform(X_test)

    # If the output is a sparse matrix, convert to dense
    if hasattr(X_test_transformed, "toarray"):
        X_test_transformed = X_test_transformed.toarray()

    # Get the feature names (e.g., Brand_Apple, Location_Colombo)
    feature_names = preprocessor.get_feature_names_out()

    # 4. SHAP TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_transformed)

    # 5. Generate and Save Summary Plot
    # This shows the global importance of each feature
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values, X_test_transformed, feature_names=feature_names, show=False
    )

    os.makedirs("reports/plots", exist_ok=True)
    plt.savefig("reports/plots/shap_summary.png", bbox_inches="tight")
    print("✅ SHAP Summary plot saved to reports/plots/shap_summary.png")

    # 6. Generate a Waterfall Plot for a single prediction
    # This shows "Why was this specific laptop priced at Rs. 150,000?"
    plt.figure(figsize=(10, 6))
    shap.plots.bar(
        explainer(X_test_transformed)[0]
    )  # Explanation for the first item in test set
    plt.savefig("reports/plots/shap_individual_explanation.png", bbox_inches="tight")
    print(
        "✅ Individual explanation plot saved to reports/plots/shap_individual_explanation.png"
    )


if __name__ == "__main__":
    explain_regression_model()
