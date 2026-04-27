import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# =====================
# إعداد الصفحة
# =====================
st.set_page_config(page_title="Coffee Forecast App", layout="wide")

st.title("☕ Coffee Demand Forecast App")
st.write("Forecast using XGBoost model + custom date selection")

# =====================
# تحميل الموديل
# =====================
model = joblib.load("xgb_model.pkl")
features = joblib.load("features.pkl")

# =====================
# تحميل البيانات
# =====================
df = pd.read_csv("data.csv")

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df.set_index('Date', inplace=True)
df = df.asfreq('D').fillna(0)

# =====================
# اختيار التاريخ
# =====================
selected_date = st.date_input("📅 اختر تاريخ التنبؤ")

selected_date = pd.to_datetime(selected_date)

# =====================
# عرض 5 أيام قبل التاريخ
# =====================
df_filtered = df.loc[:selected_date].copy()

st.subheader("📊 Last 5 Days Before Selected Date")
st.dataframe(df_filtered.tail(5))

# =====================
# زر التنبؤ
# =====================
if st.button("🔮 Predict Next 7 Days"):

    future_predictions = []
    df_future = df_filtered.copy()

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
        start=df_filtered.index[-1] + pd.Timedelta(days=1),
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
    # الرسم المحسن
    # =====================
    st.subheader("📈 Forecast Chart")

    fig, ax = plt.subplots(figsize=(12,5))

    # آخر 30 يوم قبل التاريخ
    ax.plot(df_filtered.index[-30:],
            df_filtered['Cups_Count'].iloc[-30:],
            label="Actual")

    # ربط آخر نقطة مع التوقع (بدون انقطاع)
    last_date = df_filtered.index[-1]
    last_value = df_filtered['Cups_Count'].iloc[-1]

    all_dates = [last_date] + list(forecast_df["Date"])
    all_values = [last_value] + list(forecast_df["Forecast"])

    ax.plot(all_dates, all_values,
            marker='o',
            linestyle='--',
            label="Forecast")

    # تحسين شكل التواريخ
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)

    ax.set_title("Coffee Demand Forecast (7 Days)")
    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)
