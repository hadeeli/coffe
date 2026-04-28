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
# النموذج
# =====================
model = joblib.load("xgb_model.pkl")
features = joblib.load("features.pkl")

# =====================
# البيانات
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
# Forecast engine
# =====================
def forecast_engine(df, model, features, end_date):

    df_sim = df.copy()
    current_df = df.copy()

    dates = pd.date_range(
        df.index[-1] + pd.Timedelta(days=1),
        end_date
    )

    for d in dates:

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

        current_df = pd.concat([
            current_df,
            pd.DataFrame({"Cups_Count": pred}, index=[d])
        ])

    return current_df

# =====================
# تشغيل
# =====================
if st.button("🔮 تشغيل التنبؤ"):

    final_date = selected_date + pd.Timedelta(days=n_days)

    df_sim = forecast_engine(df, model, features, final_date)

    # =====================
    # الجدول الأول (5 أيام قبل)
    # =====================
    table1 = df_sim.loc[:selected_date].iloc[:-1].tail(5).copy()

    # =====================
    # الجدول الثاني (التنبؤ)
    # =====================
    table2 = df_sim.loc[selected_date:final_date].copy()

    # =====================
    # عرض الجداول
    # =====================
    st.subheader("📊 آخر 5 أيام")
    st.dataframe(table1.rename(columns={"Cups_Count": "عدد الأكواب"}))

    st.subheader("📊 جدول التنبؤ")
    st.dataframe(table2.rename(columns={"Cups_Count": "عدد الأكواب"}))

    # =====================
    # 📈 الرسم الجديد (المطلوب)
    # =====================
    st.subheader("📈 الرسم البياني")

    fig, ax = plt.subplots(figsize=(12,5))

    # 🔵 الجدول الأول
    ax.plot(
        table1.index,
        table1["Cups_Count"],
        color="blue",
        marker="o",
        label="Table 1 (Historical 5 days)"
    )

    # 🟠 الجدول الثاني
    ax.plot(
        table2.index,
        table2["Cups_Count"],
        color="orange",
        marker="o",
        linestyle="--",
        label="Table 2 (Forecast)"
    )

    ax.axvline(selected_date, color="gray", linestyle=":")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)

    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)
