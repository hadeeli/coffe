import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Coffee Forecast App", layout="wide")

st.title("☕ Coffee Demand Forecast")
st.write("Forecast next 7 days using XGBoost model")

# تحميل النموذج
model = joblib.load("xgb_model.pkl")
features = joblib.load("features.pkl")

# تحميل البيانات
df = pd.read_excel("data.xlsx")

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df.set_index('Date', inplace=True)
df = df.asfreq('D').fillna(0)

st.subheader("📊 Raw Data")
st.dataframe(df.tail())
