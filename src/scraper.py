import time
import os
import re
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager


def scrape_ikman_electronics(pages=1):

    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()), options=chrome_options
    )

    all_products = []

    for page in range(1, pages + 1):
        print(f"Scraping Page {page}...")

        url = f"https://ikman.lk/en/ads/sri-lanka/electronics?page={page}"
        driver.get(url)

        time.sleep(6)  # wait for JS to fully load

        # 🔥 Directly get JS object
        data = driver.execute_script("return window.initialData")

        if not data:
            print("window.initialData not found.")
            continue

        try:
            ads = data["serp"]["ads"]["data"]["ads"]
            print(f"Found {len(ads)} ads on page {page}")
        except:
            print("JSON structure changed.")
            continue

        for ad in ads:
            try:
                title = ad.get("title", "Unknown")
                price_text = ad.get("price", "0")
                price = float(re.sub(r"[^\d]", "", price_text))

                location = ad.get("location", "Unknown")
                is_member = 1 if ad.get("isMember") else 0
                is_promoted = 1 if ad.get("isFeaturedAd") else 0

                # Brand extraction
                brand = "Generic"
                known_brands = [
                    "Apple",
                    "Samsung",
                    "Xiaomi",
                    "Huawei",
                    "Sony",
                    "LG",
                    "Asus",
                    "Dell",
                    "HP",
                ]

                for b in known_brands:
                    if b.lower() in title.lower():
                        brand = b
                        break

                all_products.append(
                    {
                        "Title": title,
                        "Brand": brand,
                        "Price": price,
                        "Location": location,
                        "Is_Member": is_member,
                        "Is_Promoted": is_promoted,
                        "Page": page,
                    }
                )

            except:
                continue

    driver.quit()

    df = pd.DataFrame(all_products)

    if df.empty:
        print("⚠ No data scraped.")
        return df

    # Feature Engineering
    brand_avg = df.groupby("Brand")["Price"].transform("mean")
    df["Market_Discount_Magnitude_%"] = (
        (brand_avg - df["Price"]) / brand_avg * 100
    ).round(2)

    return df


# ---------- RUN ----------
if __name__ == "__main__":

    df = scrape_ikman_electronics(pages=20)

    if df.empty:
        print("No CSV created.")
    else:
        os.makedirs("data/raw", exist_ok=True)
        df.to_csv("data/raw/ikman_market_data.csv", index=False)
        print("✅ Scraping Complete!")
        print("Saved to: data/raw/ikman_market_data.csv")
        print(df.head())
