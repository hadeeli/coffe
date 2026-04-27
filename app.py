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
# تحميل البيانات
# =====================
df = pd.read_csv("data.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")
df.set_index("Date", inplace=True)
df = df.asfreq("D").fillna(0)

last_real_date = df.index[-1]

# =====================
# Inputs
# =====================
col1, col2 = st.columns(2)

with col1:
    selected_date = st.date_input("📅 اختر تاريخ بداية التنبؤ")

with col2:
    n_days = st.number_input("📆 عدد أيام التنبؤ", 1, 30, 5, 1)

selected_date = pd.to_datetime(selected_date)

st.markdown("---")

# =====================
# Forecast Engine
# =====================
def forecast_engine(df, model, features, end_date):

    df_sim = df.copy()
    current_df = df.copy()

    all_predictions = {}
    all_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), end_date)

    for d in all_dates:

        row = pd.DataFrame(index=[d])

        row["day_of_week"] = d.dayofweek
        row["month"] = d.month
        row["is_weekend"] = 1 if d.dayofweek >= 5 else 0

        row["lag_1"] = current_df["Cups_Count"].iloc[-1]
        row["lag_7"] = current_df["Cups_Count"].iloc[-7]

        row["rolling_mean_7"] = current_df["Cups_Count"].iloc[-7:].mean()
        row["rolling_std_7"] = current_df["Cups_Count"].iloc[-7:].std()

        pred = model.predict(row[features])[0]
        pred = max(0, int(round(pred)))

        current_df = pd.concat(
            [current_df, pd.DataFrame({"Cups_Count": pred}, index=[d])]
        )

        all_predictions[d] = pred

    return current_df, all_predictions


# =====================
# تشغيل التنبؤ
# =====================
if st.button("🔮 تشغيل التنبؤ"):

    final_date = selected_date + pd.Timedelta(days=n_days)

    df_sim, preds = forecast_engine(df, model, features, final_date)

    # =====================
    # تحديد 5 أيام قبل التاريخ المختار
    # =====================
    df_sim["Type"] = np.where(df_sim.index <= last_real_date, "Historical", "Forecast")

    past_5 = df_sim.loc[:selected_date].tail(5).copy()

    # إذا الأيام كانت forecast نوضحها
    past_5["Type"] = np.where(
        past_5.index <= last_real_date,
        "Historical",
        "Forecast"
    )

    # =====================
    # جدول 1: آخر 5 أيام
    # =====================
    st.subheader("📊 آخر 5 أيام")

    st.dataframe(
        past_5[["Cups_Count", "Type"]]
        .rename(columns={
            "Cups_Count": "عدد الأكواب",
            "Type": "نوع البيانات"
        })
    )

    # =====================
    # جدول 2: التنبؤ
    # =====================
    forecast_df = df_sim.loc[selected_date:final_date].copy()

    forecast_df["Type"] = "Forecast"

    st.subheader("📊 جدول التنبؤ")

    st.dataframe(
        forecast_df[["Cups_Count", "Type"]]
        .rename(columns={
            "Cups_Count": "عدد الأكواب",
            "Type": "نوع البيانات"
        })
    )

    # =====================
    # الرسم
    # =====================
    st.subheader("📈 الرسم البياني")

    fig, ax = plt.subplots(figsize=(12,5))

    hist = df_sim.loc[:selected_date]
    fc = df_sim.loc[selected_date:]

    ax.plot(hist.index,
            hist["Cups_Count"],
            label="Historical",
            color="blue")

    ax.plot(fc.index,
            fc["Cups_Count"],
            label="Forecast",
            color="orange",
            linestyle="--")

    ax.axvline(selected_date, color="gray", linestyle=":")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)

    ax.set_title("Coffee Forecast System")
    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)
