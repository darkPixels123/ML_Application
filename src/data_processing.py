import pandas as pd
import numpy as np
import os


def process_data(path):
    # 1. Load Data
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return None

    df = pd.read_csv(path)

    # 2. Cleaning: Handle Outliers (Crucial for Regression)
    # Remove items with prices that are clearly placeholders
    df = df[df["Price"] > 500]
    df = df[df["Price"] < 2000000]

    # 3. Handle Missing Values
    df["Brand"] = df["Brand"].fillna("Generic")
    df["Location"] = df["Location"].fillna("Other")
    df["Title"] = df["Title"].fillna(
        "Unknown"
    )  # Added to prevent errors in engineering

    # 4. Final DataFrame Selection
    # CRITICAL CHANGE: Include "Title" in the selection so feature_engineering.py can use it
    required_columns = [
        "Brand",
        "Location",
        "Is_Member",
        "Is_Promoted",
        "Price",
        "Title",
    ]
    df_cleaned = df[required_columns].copy()

    print(
        f"✅ Data processing complete. {len(df_cleaned)} rows ready for feature engineering."
    )

    # Return the dataframe including 'Title'
    return df_cleaned


if __name__ == "__main__":
    process_data("./data/raw/ikman_market_data.csv")
