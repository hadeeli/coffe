import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

st.title("☕ Coffee Demand Forecast App")

st.write("7-Day Forecast using XGBoost Model")

# نفترض عندك df جاهز
df = pd.read_csv("data.csv", parse_dates=["Date"], index_col="Date")

features = [
    'lag_1', 'lag_7',
    'rolling_mean_7', 'rolling_std_7',
    'day_of_week', 'month', 'is_weekend'
]

target = "Cups_Count"
