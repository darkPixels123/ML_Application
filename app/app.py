import streamlit as st
import joblib
import pandas as pd

# Load the pipeline
model = joblib.load("models/model.pkl")

st.title("Ikman.lk Electronics Price Checker")

# User Inputs
brand = st.selectbox("Brand", ["Apple", "Samsung", "Generic", "HP", "Dell", "Sony"])
location = st.text_input("Location (e.g., Colombo, Gampaha)", "Colombo")
price = st.number_input("Listing Price (Rs.)", min_value=1000)
is_member = st.checkbox("Is the seller a Member?")
is_promoted = st.checkbox("Is this a Featured/Promoted Ad?")

if st.button("Check if Overpriced"):
    # Create a dataframe matching the features in X
    input_data = pd.DataFrame(
        [[brand, price, location, int(is_member), int(is_promoted)]],
        columns=["Brand", "Price", "Location", "Is_Member", "Is_Promoted"],
    )

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("🚨 This item is likely OVERPRICED compared to the market average.")
    else:
        st.success("✅ This seems like a FAIR MARKET PRICE.")
