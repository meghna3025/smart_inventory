import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from forecast import load_data, preprocess_data, train_model

st.set_page_config(page_title="Inventory Dashboard", layout="wide")

# -----------------------------
# UI STYLE
# -----------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #1f2937, #374151);
}
label, p, div {
    color: #F9FAFB !important;
}
div[data-baseweb="select"] {
    background-color: #FFFFFF !important;
    border-radius: 8px !important;
}
div[data-baseweb="select"] div,
div[data-baseweb="select"] input {
    color: black !important;
    font-weight: 600 !important;
}
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    padding: 10px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.title("📦 Inventory Intelligence Dashboard")
st.markdown("### Smart Demand Forecasting & Inventory Optimization")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def get_data():
    return load_data()

@st.cache_data
def get_processed():
    return preprocess_data(load_data())

raw_df = get_data()
df = get_processed()

# -----------------------------
# CLEAN DATA
# -----------------------------
raw_df['Quantity'] = pd.to_numeric(raw_df['Quantity'], errors='coerce')
raw_df['InvoiceDate'] = pd.to_datetime(raw_df['InvoiceDate'], dayfirst=True, errors='coerce')

raw_df = raw_df.dropna(subset=['InvoiceDate'])
raw_df = raw_df[raw_df['Quantity'] > 0]

# -----------------------------
# PRODUCT SELECTION
# -----------------------------
products = sorted(raw_df['Description'].dropna().unique())
product = st.selectbox("🛒 Select Product", products)

# -----------------------------
# PRODUCT DATA
# -----------------------------
product_df = raw_df[raw_df['Description'] == product]

product_df = product_df.groupby('InvoiceDate')['Quantity'].sum().reset_index()
product_df.rename(columns={'InvoiceDate': 'Date', 'Quantity': 'Sales'}, inplace=True)

product_df['lag_1'] = product_df['Sales'].shift(1)
product_df = product_df.dropna()

if len(product_df) < 10:
    st.warning("⚠️ Not enough data for this product")
    st.stop()

# -----------------------------
# PRODUCT MODEL
# -----------------------------
from sklearn.ensemble import RandomForestRegressor

X = product_df[['lag_1']]
y = product_df['Sales']

model_p = RandomForestRegressor(n_estimators=50, random_state=42)
model_p.fit(X, y)

pred = model_p.predict(product_df[['lag_1']].iloc[-1:])[0]

# -----------------------------
# INVENTORY LOGIC
# -----------------------------
std_dev = product_df['Sales'].std()
safety_stock = 1.65 * std_dev
lead_time = 2

suggested_stock = pred + safety_stock
reorder_level = (pred * lead_time) + safety_stock

# -----------------------------
# GLOBAL MODEL
# -----------------------------
model, df, features, rf_rmse, xgb_rmse, model_name = train_model(df)

# -----------------------------
# KPI SECTION
# -----------------------------
st.markdown("## 📊 Key Metrics")

c1, c2, c3 = st.columns(3)
c1.metric("📈 Predicted Sales", int(pred))
c2.metric("📦 Recommended Stock", int(suggested_stock))
c3.metric("🔁 Reorder Level", int(reorder_level))

# -----------------------------
# MODEL PERFORMANCE
# -----------------------------
st.markdown("## 🤖 Model Performance")

m1, m2, m3 = st.columns(3)
m1.metric("🌳 RF RMSE", round(rf_rmse, 2))
m2.metric("⚡ XGB RMSE", round(xgb_rmse, 2))
m3.success(f"🏆 Best Model: {model_name}")

# -----------------------------
# SMALL RMSE GRAPH
# -----------------------------
st.markdown("## 📊 Model Comparison")

col1, col2 = st.columns([1, 2])

with col1:
    fig, ax = plt.subplots(figsize=(2.5, 2))
    ax.bar(["RF", "XGB"], [rf_rmse, xgb_rmse],
           color=["#60A5FA", "#34D399"])
    ax.set_title("RMSE", fontsize=9)
    ax.tick_params(axis='both', labelsize=8)

    st.pyplot(fig)

# -----------------------------
# STATUS
# -----------------------------
st.markdown("## 🚦 Inventory Status")

if pred > 500:
    st.error("🔴 High Demand: Restock immediately")
elif pred < 100:
    st.warning("🟡 Low Demand: Overstock risk")
else:
    st.success("🟢 Inventory is balanced")

# -----------------------------
# INSIGHTS
# -----------------------------
st.markdown("## 🧠 Insights")

st.info(f"""
• Product: **{product}**

• Expected sales tomorrow: **{int(pred)} units**  
• Recommended stock: **{int(suggested_stock)} units**  
• Restock below: **{int(reorder_level)} units**

• Best model used: **{model_name}**
""")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("📌 AI-Based Inventory Optimization Dashboard")