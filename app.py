import streamlit as st
import pandas as pd
import pickle
import os

# 1. UI Setup (Minimalist/Old Money Style)
st.set_page_config(page_title="Gold Shop Analytics", layout="wide")
st.title("✨ Gold Shop Inventory Command Center")

# 2. Load Data
df = pd.read_csv('cleaned_gold_stock_data.csv')

# 3. Sidebar Selection
item_selected = st.sidebar.selectbox("Select Jewelry Item", df['item_name'].unique())

# 4. Load the specific model
model_filename = f"{item_selected}_model.pkl"

if os.path.exists(model_filename):
    with open(model_filename, 'rb') as f:
        model = pickle.load(f)
    
    # Forecast 30 days
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    
    # 5. Display Dashboard
    st.subheader(f"Demand Forecast for {item_selected}")
    st.line_chart(forecast.set_index('ds')['yhat'].tail(30))
    
    # KPI Cards
    st.metric("Predicted Demand (Next 30 Days)", int(forecast['yhat'].tail(30).sum()))
else:
    st.error("Model file not found!")