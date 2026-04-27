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

st.markdown("<h1 style='text-align:center;'>☕ نظام التنبؤ باستهلاك القهوة</h1>", unsafe_allow_html=True)

# =====================
# تحميل النموذج
# =====================
model = joblib.load("xgb_model.pkl")
features = joblib.load("features.pkl")

# =====================
# البيانات
# =====================
df = pd.read_csv("data.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df.set_index('Date', inplace=True)
df = df.asfreq('D').fillna(0)

# =====================
# INPUTS
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
# آخر 5 أيام (جدول 1)
# =====================
df_display = df.copy()
df_display["Type"] = "Historical"

st.subheader("📊 آخر 5 أيام (بيانات فعلية)")

st.dataframe(
    df_display.tail(5)[["Cups_Count"]]
    .assign(Type="Historical")
    .rename(columns={"Cups_Count": "عدد الأكواب"})
)

# =====================
# زر التنفيذ
# =====================
if st.button("🔮 تشغيل التنبؤ"):

    df_future = df.copy()
    predictions = []
    dates = []
    types = []

    last_date = df.index[-1]

    total_days = n_days

    # =====================
    # Forecast loop
    # =====================
    for i in range(total_days):

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

        predictions.append(pred)
        dates.append(next_date)
        types.append("Forecast")

    # =====================
    # FORECAST TABLE (جدول 2)
    # =====================
    forecast_df = pd.DataFrame({
        "التاريخ": dates,
        "عدد الأكواب": predictions,
        "النوع": types
    })

    st.subheader("📊 جدول التنبؤ")

    st.dataframe(forecast_df.rename(columns={
        "النوع": "نوع البيانات"
    }))

    # =====================
    # COMBINED DATA FOR PLOT
    # =====================
    df_plot = df.copy()
    df_plot["Type"] = "Historical"

    forecast_series = pd.DataFrame({
        "Cups_Count": predictions,
        "Type": "Forecast"
    }, index=dates)

    combined = pd.concat([df_plot, forecast_series])

    # =====================
    # رسم بياني
    # =====================
    st.subheader("📈 الرسم البياني")

    fig, ax = plt.subplots(figsize=(12,5))

    # قبل التنبؤ
    hist = combined[combined["Type"] == "Historical"]
    ax.plot(hist.index, hist["Cups_Count"],
            label="Historical",
            color="blue")

    # التنبؤ
    fc = combined[combined["Type"] == "Forecast"]
    ax.plot(fc.index, fc["Cups_Count"],
            label="Forecast",
            color="orange",
            linestyle="--")

    # نقطة البداية
    ax.axvline(selected_date, color="gray", linestyle=":")

    # تنسيق
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)

    ax.set_title("Coffee Demand Forecast")
    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)
