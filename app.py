import streamlit as st
import pandas as pd
import pickle
import os

# ==============================================================================
# 1. PAGE CONFIG & AESTHETIC (Old Money Minimalist)
# ==============================================================================
st.set_page_config(page_title="Gold Shop Analytics Dashboard", page_icon="💎", layout="wide")

st.markdown("""
    <style>
    /* Soft Champagne Background */
    .stApp { background-color: #F8F5F2; }
    
    /* Elegant Headers */
    h1, h2, h3 { 
        font-family: 'Georgia', serif; 
        color: #2C2C2C; 
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    /* Premium Metric Cards */
    [data-testid="stMetric"] { 
        background-color: #FFFFFF; 
        padding: 20px; 
        border-radius: 0px; 
        border-left: 4px solid #C5A059; /* Gold Accent */
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
    }
    
    /* Clean Divider */
    hr { border-top: 1px solid #D3C5A1 !important; }
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 2. LOGIC FUNCTIONS
# ==============================================================================

@st.cache_data
def load_data():
    df = pd.read_csv("data/cleaned_gold_stock_data.csv")
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["quantity_sold"] * df["unit_price"]
    return df

@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def get_inventory_status(stock):
    if stock < 20: return "🔴 Critical"
    elif stock < 50: return "🟡 Low"
    else: return "🟢 Healthy"

def get_restock_recommendation(stock, forecast_df):
    predicted_demand = int(forecast_df["yhat"].tail(30).sum())
    if stock < predicted_demand:
        return True, predicted_demand
    return False, predicted_demand

# ==============================================================================
# 3. MAIN DASHBOARD EXECUTION
# ==============================================================================

df = load_data()
st.title("💎 Gold Shop Inventory Dashboard")

# Get only the very last recorded day for each unique item
latest_inventory_df = df.sort_values("date").drop_duplicates(subset=["item_name"], keep="last")
real_inventory_value = (latest_inventory_df["stock_left"] * latest_inventory_df["unit_price"]).sum()

# KPI Section
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Inventory Value", f"{real_inventory_value:,.0f} MMK")
c2.metric("Avg Daily Sales", f"{df['quantity_sold'].mean():.1f}")
c3.metric("Items Monitored", df["item_name"].nunique())
c4.metric("Best Seller", df.groupby("item_name")["quantity_sold"].sum().idxmax())

st.divider()

# Selection & Layout
selected_item = st.selectbox("Choose Jewelry Item", sorted(df["item_name"].unique()))
left, right = st.columns([2, 1])

# Forecast Logic
model_path = os.path.join("models", f"{selected_item}_model.pkl")

with left:
    st.subheader(f"📈 Demand Forecast - {selected_item}")
    if os.path.exists(model_path):
        model = load_model(model_path)
        forecast = model.predict(model.make_future_dataframe(periods=30))
        st.line_chart(forecast.set_index("ds")["yhat"].tail(30))
    else:
        st.warning("Model file not found in /models folder.")

# Inventory & Restock Logic
with right:
    st.subheader("📦 Inventory Status")
    latest = df[df["item_name"] == selected_item].iloc[-1]
    
    st.write(f"**Current Stock:** {latest['stock_left']}")
    st.write(f"**Status:** {get_inventory_status(latest['stock_left'])}")
    
    if os.path.exists(model_path):
        is_critical, demand = get_restock_recommendation(latest['stock_left'], forecast)
        st.divider()
        st.subheader("🚚 Restock Recommendation")
        if is_critical:
            st.error(f"⚠️ Restock needed! (Forecasted demand: {demand})")
        else:
            st.success(f"✅ Stock sufficient for demand ({demand}).")

st.divider()
st.subheader("📊 Analytics Overview")
col1, col2 = st.columns(2)
col1.bar_chart(df.groupby("item_name")["quantity_sold"].sum().sort_values(ascending=False))
col2.line_chart(df.groupby("date")["quantity_sold"].sum())