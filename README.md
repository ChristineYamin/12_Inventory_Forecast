# 💎 Gold Boutique Executive Dashboard & Forecasting System
**Project 12 of "23 Projects at 23"** | *Machine Learning & Business Intelligence*
=for-the-badge&logo=pandas&logoColor=white)

## 📌 Executive Summary
Managing a high-value retail business (like a family gold shop) requires precision. Dead stock locks up valuable capital, while stockouts result in lost revenue. This project bridges the gap between traditional retail and modern data science by deploying a **Predictive Inventory Management System**. 

Using Meta's `Prophet` forecasting model, this dashboard provides 30-day demand projections, dynamic profitability tracking, and an automated 3-tier restock intelligence system, all wrapped in a luxury-grade, minimalist executive UI.

## ✨ Core Features
* **📈 Predictive Analytics:** Utilizes `Prophet` time-series forecasting to predict specific item demand for the next 30 days based on historical sales data.
* **💰 Dynamic Profitability Tracking:** Calculates estimated profit marginas dynamically via an interactive executive slider, shifting the focus from gross revenue to actual asset profitability.
* **🚨 Smart Supply Chain Logic:** A 3-tier inventory status engine (`Urgent Restock`, `Plan Restock`, `Safe`) that cross-references current stock against machine learning demand projections.
* **📊 Automated Business Insights:** Generates a real-time executive text summary of market trends and top revenue drivers.
* **🏛️ Boutique UI/UX:** A custom-styled Streamlit interface utilizing an "Old Money" aesthetic (Champagne background, Serif typography, Gold accents) tailored for high-end retail management.

## Technical  Highlights
Decoupled Architecture: Machine learning models are trained and serialized (.pkl) offline, ensuring the Streamlit presentation layer loads instantly without retraining models on the fly.

Data Robustness: Implemented .fillna(0) and try/except blocks within the UI layer to prevent application crashes during zero-sales periods or missing data points.

State Management: Utilized @st.cache_data and @st.cache_resource decorators to optimize memory usage and eliminate redundant data/model loading upon user interaction.