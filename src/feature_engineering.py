import pandas as pd
import numpy as np


def apply_feature_engineering(df):
    """
    Transforms raw scraped data into features for the Regression model.
    """
    # 1. Create Brand-based statistics
    # This gives the model a reference point for what a 'normal' price is for that brand
    brand_means = df.groupby("Brand")["Price"].transform("mean")
    df["Brand_Avg_Price"] = brand_means

    # 2. Price Relative to Brand Average
    # (Helpful for the model to see if an item is a 'deal' or 'premium')
    df["Price_Ratio_to_Brand"] = df["Price"] / brand_means

    # 3. Text-based features from Title
    # Check for keywords that typically increase/decrease value
    df["Has_Warranty"] = (
        df["Title"].str.contains("warranty|warenty", case=False).astype(int)
    )
    df["Is_Brand_New"] = (
        df["Title"].str.contains("brand new|unopened", case=False).astype(int)
    )

    # 4. Location Popularity
    # Count how many ads are in each location to represent market activity
    loc_counts = df.groupby("Location")["Price"].transform("count")
    df["Location_Ad_Density"] = loc_counts

    # 5. Clean up
    # Drop the 'Title' as we've extracted its value and it's too unique for RF
    df = df.drop(columns=["Title", "Page"], errors="ignore")

    print(
        "✅ Feature Engineering Complete: Added Brand Stats, Warranty flags, and Ad Density."
    )
    return df


if __name__ == "__main__":
    # Test the script
    raw_data = pd.read_csv("data/raw/ikman_market_data.csv")
    engineered_data = apply_feature_engineering(raw_data)
    engineered_data.to_csv("data/processed/engineered_data.csv", index=False)
