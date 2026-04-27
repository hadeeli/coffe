import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# =====================
# إعداد الصفحة
# =====================
st.set_page_config(page_title="توقع القهوة", layout="wide")

st.markdown("<h1 style='text-align:center;'>☕ نظام التنبؤ باستهلاك القهوة</h1>", unsafe_allow_html=True)

st.markdown("---")

# =====================
# تحميل النموذج
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
# Inputs
# =====================
selected_date = st.date_input("📅 اختر تاريخ بداية التنبؤ")
selected_date = pd.to_datetime(selected_date)

n_days = st.number_input(
    "📆 عدد أيام التنبؤ",
    min_value=1,
    max_value=30,
    value=5,
    step=1
)

st.markdown("---")

# =====================
# بيانات قبل التاريخ المختار
# =====================
df_before = df.loc[:selected_date].copy()

df_before["day_name"] = df_before.index.day_name()

# =====================
# 📊 جدول 1: آخر 5 أيام (يتغير حسب التاريخ)
# =====================
st.subheader("📊 آخر 5 أيام قبل التاريخ المختار")

st.dataframe(
    df_before.tail(5)[["day_name", "Cups_Count"]]
    .rename(columns={
        "day_name": "اسم اليوم",
        "Cups_Count": "عدد الأكواب"
    })
)

# =====================
# زر التنفيذ
# =====================
if st.button("🔮 تنفيذ التنبؤ"):

    df_future = df_before.copy()

    preds = []
    dates = []

    # =====================
    # Forecast loop
    # =====================
    for i in range(int(n_days)):

        next_date = df_future.index[-1] + pd.Timedelta(days=1)

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

        preds.append(pred)
        dates.append(next_date)

    # =====================
    # 📊 جدول 2: التنبؤ
    # =====================
    forecast_df = pd.DataFrame({
        "التاريخ": dates,
        "اسم اليوم": [d.day_name() for d in dates],
        "عدد الأكواب": preds
    })

    st.subheader("📊 جدول التنبؤ")

    st.dataframe(forecast_df)

    # =====================
    # 📈 الرسم (مُصلح بالكامل)
    # =====================
    st.subheader("📈 الرسم البياني")

    fig, ax = plt.subplots(figsize=(12,5))

    # 🔵 آخر 30 يوم فقط قبل التاريخ
    hist = df.loc[:selected_date].tail(30)

    ax.plot(hist.index,
            hist["Cups_Count"],
            label="Historical",
            color="blue")

    # 🟠 التنبؤ
    ax.plot(dates,
            preds,
            label="Forecast",
            color="orange",
            linestyle="--")

    # خط بداية التنبؤ
    ax.axvline(selected_date, color="gray", linestyle=":")

    # تحسين التاريخ
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)

    ax.set_title("Coffee Demand Forecast")
    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)
