import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Configuration
st.set_page_config(
    page_title="HousePrice AI",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional/Vibrant Look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f1f5f9;
    }
    
    .main-header {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-header h1 {
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 3rem;
    }
    
    .prediction-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        padding: 2rem;
        border-radius: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        text-align: center;
    }
    
    .price-value {
        font-size: 3.5rem;
        font-weight: 700;
        color: #10b981;
        margin-top: 1rem;
    }
    
    .stNumberInput label {
        color: #94a3b8 !important;
        font-weight: 600 !important;
    }
    
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        color: white;
        font-weight: 700;
        padding: 0.75rem;
        border-radius: 0.75rem;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(56, 189, 248, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_assets():
    model = joblib.load("model.pkl")
    features = joblib.load("features.pkl")
    return model, features

model, features = load_assets()

# Sidebar for additional info or settings
with st.sidebar:
    st.image("https://img.icons8.com/color/144/home--v1.png", width=100)
    st.title("Settings")
    st.write("Welcome to HousePrice AI. Use the sliders or inputs to predict the market value of your property.")
    st.divider()
    st.markdown("### Model Stats")
    st.info("Linear Regression Model trained on 5,000 recent sales data.")

# Main Header
st.markdown("""
<div class="main-header">
    <h1>HousePrice AI Predictor</h1>
    <p style="color: #94a3b8; font-size: 1.2rem;">Expert house valuation powered by machine learning</p>
</div>
""", unsafe_allow_html=True)

# Inputs and Predictions
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.subheader("Property Features")
    
    c1, c2 = st.columns(2)
    with c1:
        avg_area_income = st.number_input("Average Area Income (₹)", min_value=10000.0, max_value=200000.0, value=68000.0, step=1000.0)
        avg_area_house_age = st.number_input("Average House Age", min_value=0.0, max_value=20.0, value=6.0, step=0.1)
    
    with c2:
        avg_area_rooms = st.number_input("Average Number of Rooms", min_value=1.0, max_value=15.0, value=7.0, step=0.1)
        avg_area_bedrooms = st.number_input("Average Number of Bedrooms", min_value=1.0, max_value=10.0, value=4.0, step=0.1)
        
    area_population = st.number_input("Area Population", min_value=100.0, max_value=100000.0, value=36000.0, step=100.0)

with col2:
    st.subheader("Valuation")
    st.write("Click predict to see the estimated value of the property based on local market trends.")
    
    # Prediction logic
    input_data = pd.DataFrame([[avg_area_income, avg_area_house_age, avg_area_rooms, avg_area_bedrooms, area_population]], columns=features)
    
    if st.button("Calculate Value"):
        prediction = model.predict(input_data)[0]
        st.markdown(f"""
        <div class="prediction-card">
            <h3 style="margin-bottom: 0;">Estimated Value</h3>
            <div class="price-value">₹{prediction:,.2f}</div>
            <p style="color: #94a3b8; font-size: 0.9rem; margin-top: 1rem;">*Based on current market simulation</p>
        </div>
        """, unsafe_allow_html=True)
        st.balloons()
    else:
        st.markdown("""
        <div class="prediction-card">
            <h3 style="color: #64748b;">Ready to Predict</h3>
            <p>Adjust the values and click the button</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #64748b; padding-bottom: 2rem;">
    Powered by HousePrice AI & Antigravity Assistant © 2026
</div>
""", unsafe_allow_html=True)
