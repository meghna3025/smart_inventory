# 📦 Smart Inventory Intelligence Dashboard

## Overview

Smart Inventory Intelligence Dashboard is a Machine Learning-powered inventory forecasting system that helps businesses predict product demand, optimize stock levels, and make data-driven inventory decisions.

The application analyzes historical sales data from retail transactions and uses machine learning models to forecast future demand while providing actionable inventory recommendations.

---

## Features

### 📈 Demand Forecasting

* Predicts future product sales using historical transaction data.
* Supports product-specific demand analysis.

### 🤖 Machine Learning Models

* Random Forest Regressor
* XGBoost Regressor
* Automatic selection of the best-performing model based on RMSE.

### 📦 Inventory Optimization

* Recommended stock calculation
* Safety stock estimation
* Reorder level calculation
* Inventory risk assessment

### 📊 Interactive Dashboard

* Product selection interface
* Key performance metrics
* Model performance comparison
* Demand insights and recommendations

### 🚦 Inventory Status Monitoring

* High-demand alerts
* Overstock risk warnings
* Balanced inventory recommendations

---

## Tech Stack

### Frontend

* Streamlit

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* XGBoost
* Random Forest

### Visualization

* Matplotlib

---

## Project Structure

```text
SMART_INVENTORY_PROJECT/
│
├── app.py
├── forecast.py
├── requirements.txt
├── data/
│   └── Online Retail.csv
├── models/
│   ├── rf_model.pkl
│   └── xgb_model.pkl
└── .gitignore
```

---

## Dataset

The project uses the Online Retail Dataset containing:

* Product descriptions
* Sales quantities
* Invoice dates
* Customer transaction information

Data is cleaned and transformed into time-series sales records for forecasting.

---

## Machine Learning Workflow

### Data Preprocessing

* Handle missing values
* Convert dates and quantities
* Aggregate sales by date
* Generate lag features
* Calculate rolling averages
* Extract calendar features

### Feature Engineering

* Lag-1 sales
* Lag-2 sales
* Rolling mean
* Day of month
* Month
* Weekday

### Model Evaluation

Two models are trained and compared:

1. Random Forest Regressor
2. XGBoost Regressor

Evaluation Metric:

* RMSE (Root Mean Squared Error)

The model with the lowest RMSE is selected as the best forecasting model.

---

## Inventory Optimization Logic

### Safety Stock

Safety stock is calculated using demand variability.

### Recommended Stock

Recommended Stock = Forecasted Demand + Safety Stock

### Reorder Level

Reorder Level = (Forecasted Demand × Lead Time) + Safety Stock

---

## Installation

### Clone Repository

```bash
git clone https://github.com/meghna3025/smart_inventory.git
cd smart_inventory
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

## Dashboard Outputs

* Predicted Sales
* Recommended Inventory Level
* Reorder Threshold
* Model Performance Metrics
* Product-Level Demand Insights
* Inventory Status Alerts

---

## Future Enhancements

* Multi-product forecasting
* Real-time inventory tracking
* Cloud deployment
* Automated purchase order generation
* Advanced time-series forecasting models
* Inventory cost optimization

---

## Author

Meghana V

Machine Learning | Data Analytics | Inventory Optimization | Streamlit Development
