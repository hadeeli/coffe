import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# =====================
# إعداد الصفحة
# =====================
st.set_page_config(page_title="Coffee Forecast", layout="wide")

st.markdown("<h1 style='text-align:center;'>☕ نظام التنبؤ الذكي للقهوة</h1>", unsafe_allow_html=True)

st.markdown("---")

# =====================
# تحميل النموذج
# =====================
model = joblib.load("xgb_model.pkl")
features = joblib.load("features.pkl")

# =====================
# تحميل البيانات (ثابتة)
# =====================
df = pd.read_csv("data.csv")

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df.set_index('Date', inplace=True)
df = df.asfreq('D').fillna(0)

last_real_date = df.index[-1]

# =====================
# Inputs
# =====================
col1, col2 = st.columns(2)

with col1:
    selected_date = st.date_input("📅 اختر تاريخ بداية التنبؤ")

with col2:
    n_days = st.number_input(
        "📆 عدد أيام التنبؤ",
        min_value=1,
        max_value=30,
        value=5,
        step=1
    )

selected_date = pd.to_datetime(selected_date)

st.markdown("---")

# =====================
# 📊 5 أيام قبل (للعرض فقط)
# =====================
df["day_name"] = df.index.day_name()

st.subheader("📊 آخر 5 أيام (سياق تاريخي)")

st.dataframe(
    df.tail(5)[["day_name", "Cups_Count"]]
    .rename(columns={
        "day_name": "اسم اليوم",
        "Cups_Count": "عدد الأكواب"
    })
)

# =====================
# 🔮 Forecast Engine (المهم هنا)
# =====================
if st.button("🔮 تشغيل التنبؤ"):

    df_future = df.copy()

    predictions = []
    dates = []

    # دايم يبدأ من آخر بيانات حقيقية
    current_date = last_real_date

    for i in range(n_days):

        next_date = current_date + pd.Timedelta(days=1)

        row = pd.DataFrame(index=[next_date])

        row["day_of_week"] = next_date.dayofweek
        row["month"] = next_date.month
        row["is_weekend"] = 1 if next_date.dayofweek >= 5 else 0

        row["lag_1"] = df_future["Cups_Count"].iloc[-1]
        row["lag_7"] = df_future["Cups_Count"].iloc[-7]

        row["rolling_mean_7"] = df_future["Cups_Count"].iloc[-7:].mean()
        row["rolling_std_7"] = df_future["Cups_Count"].iloc[-7:].std()

        pred = model.predict(row[features])[0]
        pred = max(0, int(round(pred)))

        df_future = pd.concat([
            df_future,
            pd.DataFrame({"Cups_Count": pred}, index=[next_date])
        ])

        predictions.append(pred)
        dates.append(next_date)

        current_date = next_date

    # =====================
    # 📊 جدول التنبؤ
    # =====================
    forecast_df = pd.DataFrame({
        "التاريخ": dates,
        "اسم اليوم": [d.day_name() for d in dates],
        "عدد الأكواب": predictions
    })

    st.subheader("📊 جدول التنبؤ")

    st.dataframe(forecast_df)

    # =====================
    # 📈 الرسم الصحيح (بدون زحمة)
    # =====================
    st.subheader("📈 الرسم البياني")

    fig, ax = plt.subplots(figsize=(12,5))

    # 🔵 آخر 30 يوم فقط
    hist = df.tail(30)

    ax.plot(hist.index,
            hist["Cups_Count"],
            label="Historical",
            color="blue")

    # 🟠 forecast
    ax.plot(dates,
            predictions,
            label="Forecast",
            color="orange",
            linestyle="--")

    # خط بداية التنبؤ
    ax.axvline(last_real_date, color="gray", linestyle=":")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)

    ax.set_title("Coffee Demand Forecast")
    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)
