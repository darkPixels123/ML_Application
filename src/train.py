import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    auc,
)

# Import your custom processing function
from src.data_processing import process_data


def train_model():
    # 1. Load data
    # Ensure data_processing.py has 'return df' at the end of process_data
    df = process_data("data/raw/ikman_market_data.csv")

    if df is None:
        print(
            "Error: data_processing.process_data returned None. Check your return statement."
        )
        return

    # 2. Define Target Variable
    # 1 if price is higher than brand average (Overpriced), 0 otherwise
    df["Overpriced"] = (df["Market_Discount_Magnitude_%"] < 0).astype(int)

    # 3. Feature Selection
    X = df[["Brand", "Price", "Location", "Is_Member", "Is_Promoted"]]
    y = df["Overpriced"]

    # 4. Define Preprocessing
    categorical_cols = ["Brand", "Location"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ],
        remainder="passthrough",
    )

    # 5. Define Model (Random Forest as per guidelines)
    model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)

    # 6. Create Pipeline
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    # 7. Train-Test Split (Guideline 3 requirement)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 8. Train the model
    print("Training the model...")
    pipeline.fit(X_train, y_train)

    # 9. Evaluate Performance
    y_pred = pipeline.predict(X_test)
    print("\n--- Model Performance ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # 10. Save Model and Test Data (Crucial for evaluate.py and app.py)
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    joblib.dump(pipeline, "models/model.pkl")
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)
    print("\n✅ Model saved to models/model.pkl")
    print("✅ Test data saved to data/processed/ for evaluation.")

    # 11. ROC Curve (Technical Clarity marks)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")

    # Save the plot for your report
    os.makedirs("notebooks", exist_ok=True)
    plt.savefig("notebooks/roc_curve.png")
    print("✅ ROC Curve saved to notebooks/roc_curve.png")
    print(f"AUC Score: {roc_auc:.4f}")


if __name__ == "__main__":
    train_model()
