import streamlit as st
import pandas as pd
import pickle
import os

# ==============================================================================
# 1. CORE FUNCTIONS (Logic & Data)
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

def get_top_sellers(df):
    return df.groupby("item_name")["quantity_sold"].sum().sort_values(ascending=False)

def get_revenue_trends(df):
    return df.groupby("date")["revenue"].sum()

# ==============================================================================
# 2. UI FUNCTIONS (Presentation)
# ==============================================================================

def render_kpis(df):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Inventory Value", f"{(df['stock_left'] * df['unit_price']).sum():,.0f} MMK")
    c2.metric("Avg Daily Sales", f"{df['quantity_sold'].mean():.1f}")
    c3.metric("Items Monitored", df["item_name"].nunique())
    c4.metric("Best Seller", df.groupby("item_name")["quantity_sold"].sum().idxmax())

def render_forecast_section(selected_item, df):
    model_path = os.path.join("models", f"{selected_item}_model.pkl")
    if os.path.exists(model_path):
        model = load_model(model_path)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        st.line_chart(forecast.set_index("ds")["yhat"].tail(30))
        return forecast
    else:
        st.warning("Model not found.")
        return None

# ==============================================================================
# 3. MAIN DASHBOARD EXECUTION
# ==============================================================================

st.set_page_config(page_title="Gold Shop Analytics", layout="wide")

# CSS Injection for Old Money Aesthetic
st.markdown("""
    <style>
    .stApp { background-color: #FAF9F6; }
    h1 { color: #2C2C2C; font-family: 'Georgia', serif; }
    </style>
    """, unsafe_allow_html=True)

df = load_data()
st.title("💎 Gold Shop Inventory Command Center")

render_kpis(df)
st.divider()

selected_item = st.selectbox("Choose Jewelry Item", sorted(df["item_name"].unique()))
left, right = st.columns([2, 1])

with left:
    st.subheader(f"📈 Demand Forecast - {selected_item}")
    forecast = render_forecast_section(selected_item, df)

with right:
    st.subheader("📦 Inventory Status")
    latest = df[df["item_name"] == selected_item].iloc[-1]
    st.write(f"**Status:** {get_inventory_status(latest['stock_left'])}")
    st.write(f"**Current Stock:** {latest['stock_left']}")

st.divider()
st.subheader("📊 Analytics Overview")
col1, col2 = st.columns(2)
with col1:
    st.write("### Top Sellers")
    st.bar_chart(get_top_sellers(df))
with col2:
    st.write("### Revenue Trends")
    st.line_chart(get_revenue_trends(df))