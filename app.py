import streamlit as st
import pandas as pd
import pickle
import os

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Gold Shop Analytics Dashboard",
    page_icon="💎",
    layout="wide"
)

# =========================
# LOAD DATA
# =========================

@st.cache_data
def load_data():
    df = pd.read_csv("data/cleaned_gold_stock_data.csv")
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["quantity_sold"] * df["unit_price"]
    return df


# =========================
# INVENTORY STATUS
# =========================

def get_inventory_status(stock):

    if stock < 20:
        return "🔴 Critical"
    elif stock < 50:
        return "🟡 Low"
    else:
        return "🟢 Healthy"


# =========================
# LOAD MODEL
# =========================

@st.cache_resource
def load_model(model_path):
    with open(model_path, "rb") as f:
        return pickle.load(f)


# =========================
# MAIN APP
# =========================

df = load_data()

st.title("💎 Gold Shop Inventory Command Center")

# =========================
# KPI SECTION
# =========================

inventory_value = (df["stock_left"] * df["unit_price"]).sum()
avg_daily_sales = df["quantity_sold"].mean()
items_monitored = df["item_name"].nunique()

best_seller = (
    df.groupby("item_name")["quantity_sold"]
    .sum()
    .idxmax()
)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Inventory Value", f"{inventory_value:,.0f} MMK")
c2.metric("Avg Daily Sales", f"{avg_daily_sales:.1f}")
c3.metric("Items Monitored", items_monitored)
c4.metric("Best Seller", best_seller)

st.divider()

# =========================
# ITEM SELECTION
# =========================

st.subheader("🔍 Select Product")

selected_item = st.selectbox(
    "Choose Jewelry Item",
    sorted(df["item_name"].unique())
)

item_df = df[df["item_name"] == selected_item]

latest_record = item_df.sort_values("date").iloc[-1]

stock_left = latest_record["stock_left"]
status = get_inventory_status(stock_left)

model_path = os.path.join("models", f"{selected_item}_model.pkl")

# =========================
# LAYOUT
# =========================

left, right = st.columns([2, 1])

# =========================
# FORECAST
# =========================

with left:

    st.subheader(f"📈 Demand Forecast - {selected_item}")

    if os.path.exists(model_path):

        try:
            model = load_model(model_path)

            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)

            chart_df = forecast[["ds", "yhat"]].tail(30)
            chart_df = chart_df.set_index("ds")

            st.line_chart(chart_df)

            forecast_export = forecast[["ds", "yhat"]].tail(30)

            csv = forecast_export.to_csv(index=False)

            st.download_button(
                label="📥 Download Forecast CSV",
                data=csv,
                file_name=f"{selected_item}_forecast.csv",
                mime="text/csv"
            )

            next_month_demand = int(forecast["yhat"].tail(30).sum())

            st.info(
                f"Predicted demand for next 30 days: {next_month_demand}"
            )

        except Exception as e:
            st.error(f"Error loading model: {e}")

    else:
        st.warning(f"No model found for {selected_item}")


# =========================
# INVENTORY DETAILS
# =========================

with right:

    st.subheader("📦 Inventory Status")

    st.write(f"**Current Stock:** {stock_left}")
    st.write(f"**Status:** {status}")
    st.write(f"**Unit Price:** {latest_record['unit_price']:,.0f} MMK")
    st.write(f"**Category:** {latest_record['category']}")


# =========================
# RESTOCK LOGIC
# =========================

st.divider()

st.subheader("🚚 Restock Recommendation")

if os.path.exists(model_path):

    try:

        forecast_demand = int(forecast["yhat"].tail(30).sum())

        if stock_left < forecast_demand:
            st.error(
                f"⚠️ Stock ({stock_left}) is lower than forecasted demand ({forecast_demand}). Restock recommended!"
            )
        else:
            st.success(
                f"✅ Stock is sufficient for forecasted demand."
            )

    except:
        pass


# =========================
# REVENUE ANALYSIS
# =========================

st.divider()

st.subheader("💰 Revenue by Item")

revenue_by_item = (
    df.groupby("item_name")["revenue"]
    .sum()
    .sort_values(ascending=False)
)

st.bar_chart(revenue_by_item)


# =========================
# TOP SELLERS
# =========================

st.divider()

st.subheader("🏆 Top Selling Items")

top_items = (
    df.groupby("item_name")["quantity_sold"]
    .sum()
    .sort_values(ascending=False)
)

st.bar_chart(top_items)


# =========================
# LOW STOCK ALERT
# =========================

st.divider()

st.subheader("🚨 Low Stock Items")

latest_stock = (
    df.sort_values("date")
    .groupby("item_name")
    .tail(1)
)

low_stock = latest_stock[
    latest_stock["stock_left"] < 20
][["item_name", "stock_left"]]

if len(low_stock) > 0:
    st.dataframe(low_stock)
else:
    st.success("No low stock items detected.")


# =========================
# SALES TREND
# =========================

st.divider()

st.subheader("📊 Sales Trend Over Time")

daily_sales = (
    df.groupby("date")["quantity_sold"]
    .sum()
)

st.line_chart(daily_sales)