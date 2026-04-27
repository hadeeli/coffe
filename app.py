import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# =====================
# إعداد الصفحة
# =====================
st.set_page_config(page_title="Coffee Forecast App", layout="wide")

st.title("☕ Coffee Demand Forecast App")
st.write("Forecast next 7 days using XGBoost model")

# =====================
# تحميل الموديل
# =====================
model = joblib.load("xgb_model.pkl")
features = joblib.load("features.pkl")

# =====================
# تحميل البيانات (CSV)
# =====================
df = pd.read_csv("data.csv")

# تحويل التاريخ
df['Date'] = pd.to_datetime(df['Date'])

# ترتيب + Index
df = df.sort_values('Date')
df.set_index('Date', inplace=True)

# إعادة ضبط التردد اليومي
df = df.asfreq('D').fillna(0)

# =====================
# عرض البيانات
# =====================
st.subheader("📊 Data Preview")
st.dataframe(df.tail())

# =====================
# زر التنبؤ
# =====================
if st.button("🔮 Predict Next 7 Days"):

    future_predictions = []
    df_future = df.copy()

    # =====================
    # Forecast loop
    # =====================
    for i in range(7):

        next_date = df_future.index[-1] + pd.Timedelta(days=1)

        new_row = pd.DataFrame(index=[next_date])

        new_row['day_of_week'] = next_date.dayofweek
        new_row['month'] = next_date.month
        new_row['is_weekend'] = 1 if next_date.dayofweek >= 5 else 0

        new_row['lag_1'] = df_future['Cups_Count'].iloc[-1]
        new_row['lag_7'] = df_future['Cups_Count'].iloc[-7]

        new_row['rolling_mean_7'] = df_future['Cups_Count'].iloc[-7:].mean()
        new_row['rolling_std_7'] = df_future['Cups_Count'].iloc[-7:].std()

        pred = model.predict(new_row[features])[0]
        pred = max(0, int(round(pred)))

        future_predictions.append(pred)

        new_row['Cups_Count'] = pred
        df_future = pd.concat([df_future, new_row])

    # =====================
    # إنشاء التواريخ
    # =====================
    future_dates = pd.date_range(
        start=df.index[-1] + pd.Timedelta(days=1),
        periods=7
    )

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Forecast": future_predictions
    })

    # =====================
    # عرض الجدول
    # =====================
    st.subheader("📅 Forecast Table")
    st.dataframe(forecast_df)

    # =====================
    # الرسم
    # =====================
    st.subheader("📈 Forecast Chart")

    fig, ax = plt.subplots(figsize=(10,5))

    ax.plot(df.index[-30:], df['Cups_Count'].iloc[-30:], label="Actual")
    ax.plot(forecast_df["Date"], forecast_df["Forecast"],
            marker='o', linestyle='--', label="Forecast")

    ax.set_title("7-Day Coffee Demand Forecast")
    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)
