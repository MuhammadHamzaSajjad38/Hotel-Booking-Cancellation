import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode()

# ==================================================
# Page Configuration
# ==================================================
st.set_page_config(
    page_title="Hotel Booking Cancellation Predictor",
    page_icon="🏨",
    layout="wide"
)

# ==================================================
# Load Model & Assets
# ==================================================
@st.cache_resource
def load_assets():
    with open("rf_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("model_columns.pkl", "rb") as f:
        columns = pickle.load(f)

    return model, scaler, columns

model, scaler, model_columns = load_assets()

# ==================================================
# Custom CSS (Professional Dark Cards)
# ==================================================
bg_img = get_base64_image("ds_bg.jpg")

st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpg;base64,{bg_img}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}}

[data-testid="stAppViewContainer"] > .main {{
    background-color: rgba(0, 0, 0, 0.6);
}}
</style>
""", unsafe_allow_html=True)




# ==================================================
# Sidebar Navigation
# ==================================================
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "🔍 Prediction", "📈 Model Info", "📌 About Project"]
)

# ==================================================
# HOME PAGE
# ==================================================
if page == "🏠 Home":
    st.title("🏨 Hotel Booking Cancellation Prediction System")

    st.markdown("""
    ### 🎯 Objective
    Predict whether a hotel booking is **likely to be canceled** using
    machine learning based on booking and customer details.

    ### 🧠 Model Used
    - **Random Forest Classifier**
    - Trained on historical hotel booking data
    - Optimized for real-time predictions
    """)

    # ---- Metric Cards ----
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="background:#0b132b;padding:25px;border-radius:15px;text-align:center;color:white">
            <h3>Model Accuracy</h3>
            <h1>84%</h1>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background:#0b132b;padding:25px;border-radius:15px;text-align:center;color:white">
            <h3>F1-Score</h3>
            <h1>0.67</h1>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="background:#0b132b;padding:25px;border-radius:15px;text-align:center;color:white">
            <h3>Dataset Size</h3>
            <h1>119K+</h1>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📊 Booking Outcome Distribution (Sample View)")

    chart_data = pd.DataFrame({
        "Booking Status": ["Not Canceled", "Canceled"],
        "Percentage": [65, 35]
    }).set_index("Booking Status")

    st.bar_chart(chart_data)

# ==================================================
# PREDICTION PAGE
# ==================================================
elif page == "🔍 Prediction":
    st.title("🔍 Predict Booking Cancellation")

    st.markdown("Fill in the booking details below:")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            hotel = st.selectbox("Hotel Type", ["City Hotel", "Resort Hotel"])
            lead_time = st.number_input("Lead Time (days)", 0, 365, 30)
            adr = st.number_input("Average Daily Rate (ADR)", 0.0, 500.0, 100.0)

        with col2:
            total_stay = st.number_input("Total Stay Nights", 0, 30, 3)
            special_requests = st.number_input("Special Requests", 0, 5, 0)
            arrival_month = st.selectbox(
                "Arrival Month",
                ["January","February","March","April","May","June",
                 "July","August","September","October","November","December"]
            )

        with col3:
            deposit_type = st.selectbox("Deposit Type", ["No Deposit", "Non Refund"])
            customer_type = st.selectbox(
                "Customer Type",
                ["Transient", "Contract", "Transient-Party", "Group"]
            )

        submit = st.form_submit_button("🚀 Predict")

    if submit:
        month_map = {
            "January":1,"February":2,"March":3,"April":4,"May":5,"June":6,
            "July":7,"August":8,"September":9,"October":10,"November":11,"December":12
        }

        input_df = pd.DataFrame({
            "hotel": [hotel],
            "lead_time": [lead_time],
            "adr": [adr],
            "total_stay_nights": [total_stay],
            "total_of_special_requests": [special_requests],
            "arrival_date_month": [month_map[arrival_month]],
            "deposit_type": [deposit_type],
            "customer_type": [customer_type]
        })

        input_encoded = pd.get_dummies(input_df)
        input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

        input_scaled = scaler.transform(input_encoded)

        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        st.markdown("---")
        st.subheader("📊 Prediction Result")

        if pred == 1:
            st.error(f"❌ **High Cancellation Risk**  \nProbability: **{prob:.2%}**")
        else:
            st.success(f"✅ **Low Cancellation Risk**  \nProbability: **{prob:.2%}**")

        st.progress(int(prob * 100))

# ==================================================
# MODEL INFO PAGE
# ==================================================
elif page == "📈 Model Info":
    st.title("📈 Model Information")

    st.markdown("""
    ### 🔧 Algorithm
    **Random Forest Classifier**

    ### 📊 Performance
    - Accuracy: **84%**
    - F1-Score: **0.67**

    ### 🧪 Why Random Forest?
    - Handles non-linear patterns
    - Robust to noise
    - Performs well on mixed data types
    """)

    st.markdown("---")
    st.subheader("🔍 Top 10 Important Features")

    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=model_columns)\
        .sort_values(ascending=False)\
        .head(10)

    st.bar_chart(feat_imp)

# ==================================================
# ABOUT PAGE
# ==================================================
elif page == "📌 About Project":
    st.title("📌 About This Project")

    st.markdown("""
    **Course:** Tools & Techniques in Data Science  
    **Project Type:** Individual Semester Project  

    ### 📚 Technologies Used
    - Python
    - Pandas, NumPy
    - Scikit-Learn
    - Streamlit

    ### 👨‍🎓 Student
    **Muhammad Hamza Sajjad**

    ### 🎓 Description
    This project demonstrates the complete data science pipeline:
    Data Cleaning → EDA → Statistical Testing → Machine Learning → Deployment
    """)
