import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="Ikman.lk Price Predictor", page_icon="💻")


# 2. Load the trained pipeline
@st.cache_resource  # Caches the model so it doesn't reload on every click
def load_model():
    # Use the path relative to the root of your project
    return joblib.load("models/model.pkl")


model = load_model()

# 3. Sidebar for Additional Context (Guideline 7: Technical Clarity)
st.sidebar.title("About the Model")
st.sidebar.info(
    "This AI model uses a Random Forest Regressor to estimate the fair market price "
    "of electronics based on real-time data scraped from Ikman.lk."
)

# 4. Main UI
st.title("🔌 Electronics Price Estimator")
st.markdown("Enter the details of the item to get a predicted market valuation.")

# 5. User Input Form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        brand = st.selectbox(
            "Brand",
            ["Apple", "Samsung", "HP", "Dell", "Sony", "LG", "Huawei", "Generic"],
        )
        location = st.selectbox(
            "Location",
            ["Colombo", "Gampaha", "Kandy", "Kalutara", "Kurunegala", "Galle"],
        )
        is_brand_new = st.radio(
            "Condition",
            options=[0, 1],
            format_func=lambda x: "Brand New" if x == 1 else "Used",
        )

    with col2:
        is_member = st.checkbox("Seller is a Member")
        is_promoted = st.checkbox("Ad is Promoted/Featured")
        # ADDED: Has_Warranty toggle to satisfy the model requirement
        has_warranty = st.checkbox("Includes Warranty")

        brand_avg_price = st.number_input(
            "Average Price for this Brand (if known)", value=50000
        )

    submit_button = st.form_submit_button("Predict Fair Price")

# 6. Prediction Logic
if submit_button:
    # Prepare input data exactly like the training features
    # NOTE: The keys here must match the column names used in train.py EXACTLY
    input_data = pd.DataFrame(
        {
            "Brand": [brand],
            "Location": [location],
            "Is_Member": [int(is_member)],
            "Is_Promoted": [int(is_promoted)],
            "Brand_Avg_Price": [brand_avg_price],
            "Is_Brand_New": [int(is_brand_new)],
            "Has_Warranty": [int(has_warranty)],
        }
    )

    # CRITICAL: Ensure the columns are in the EXACT same order as training
    # Scikit-learn pipelines are sensitive to column positioning
    column_order = [
        "Brand",
        "Location",
        "Is_Member",
        "Is_Promoted",
        "Brand_Avg_Price",
        "Is_Brand_New",
        "Has_Warranty",
    ]
    input_data = input_data[column_order]

    try:
        # Generate Prediction
        prediction = model.predict(input_data)[0]

        # Display Result
        st.divider()
        st.subheader("Market Valuation Result")
        st.metric(label="Estimated Fair Market Price", value=f"Rs. {prediction:,.2f}")

        # Interpretation (Guideline 4: Align with domain knowledge)
        if brand == "Apple" and is_brand_new:
            st.caption(
                "ℹ️ Note: Premium brands in new condition typically hold higher resale value in the Sri Lankan market."
            )

        if prediction < 5000:
            st.warning(
                "⚠️ The predicted price is very low. Please ensure the 'Average Brand Price' input is realistic."
            )

        st.balloons()

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.info(
            "Ensure that your trained model (model.pkl) was trained with the 'Has_Warranty' feature."
        )
