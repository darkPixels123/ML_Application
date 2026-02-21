import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os


def process_data(path):
    # 1. Load Data
    if not os.path.exists(path):
        print("Raw data not found. Run the scraper first.")
        return

    df = pd.read_csv(path)

    # 2. Cleaning: Remove outliers
    # Example: Remove items with price 0 or extremely high outliers that might be errors
    df = df[df["Price"] > 100]

    # 3. Feature Selection
    # We drop 'Title' because it's unique text, and 'Page' as it's not a physical feature
    features = ["Brand", "Location", "Is_Member", "Is_Promoted"]
    target = "Price"

    X = df[features].copy()
    y = df[target].copy()

    # 4. Encoding Categorical Data (Brand and Location)
    # Random Forest works well with Label Encoding for categorical features
    le_brand = LabelEncoder()
    le_loc = LabelEncoder()

    X["Brand"] = le_brand.fit_transform(X["Brand"])
    X["Location"] = le_loc.fit_transform(X["Location"])

    # 5. Split Data
    # 80% Training, 20% Testing (Standard requirement for Guideline 3)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 6. Scaling (Optional but good practice)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 7. Save Processed Data for Training
    os.makedirs("./data/processed", exist_ok=True)
    pd.DataFrame(X_train_scaled, columns=features).to_csv(
        "./data/processed/X_train.csv", index=False
    )
    pd.DataFrame(X_test_scaled, columns=features).to_csv(
        "./data/processed/X_test.csv", index=False
    )
    y_train.to_csv("./data/processed/y_train.csv", index=False)
    y_test.to_csv("./data/processed/y_test.csv", index=False)

    print("✅ Data processing complete. Files saved in data/processed/")
    return df


if __name__ == "__main__":
    process_data()
