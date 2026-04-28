import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# =====================
# Page setup
# =====================
st.set_page_config(page_title="Coffee Forecast", layout="wide")

# 🔥 مهم: تقليل عرض الكونتينر عشان الرسم يصغر فعليًا
st.markdown("""
<style>
.stApp {
    background-color: #efe6dd;
}

/* تقليل عرض الصفحة (حل مشكلة الرسم) */
.block-container {
    max-width: 900px;
    padding-top: 2rem;
}

/* الجداول */
div[data-testid="stDataFrame"] {
    background-color: #f2f2f2;
    border-radius: 12px;
    padding: 8px;
}

/* العناوين */
h3, h4 {
    font-weight: 500 !important;
    color: #5a5a5a !important;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# =====================
# Title
# =====================
st.markdown("""
<div style="
    background:#6f4e37;
    padding:12px;
    border-radius:12px;
    text-align:center;
    color:white;">
<h2 style="color:white; font-weight:500;">
☕ نظام التنبؤ الذكي للقهوة
</h2>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# =====================
# Model
# =====================
model = joblib.load("xgb_model.pkl")
features = joblib.load("features.pkl")

# =====================
# Data
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
    n_days = st.number_input(
        "📆 عدد أيام التنبؤ",
        min_value=1,
        max_value=30,
        value=5,
        step=1,
        format="%d"
    )

selected_date = pd.to_datetime(selected_date)

st.markdown("---")

# =====================
# Forecast engine
# =====================
def forecast_engine(df, model, features, start_date, end_date):

    current = df.copy()
    dates = pd.date_range(start=start_date, end=end_date)

    for d in dates:

        row = pd.DataFrame(index=[d])

        row["day_of_week"] = d.dayofweek
        row["month"] = d.month
        row["is_weekend"] = 1 if d.dayofweek >= 5 else 0

        row["lag_1"] = current["Cups_Count"].iloc[-1]
        row["lag_7"] = current["Cups_Count"].iloc[-7]

        row["rolling_mean_7"] = current["Cups_Count"].iloc[-7:].mean()
        row["rolling_std_7"] = current["Cups_Count"].iloc[-7:].std()

        pred = model.predict(row[features])[0]
        pred = max(0, int(round(pred)))

        current = pd.concat([
            current,
            pd.DataFrame({"Cups_Count": pred}, index=[d])
        ])

    return current

# =====================
# Run
# =====================
if st.button("🔮 تشغيل التنبؤ"):

    if selected_date <= last_real_date:
        start_forecast = selected_date + pd.Timedelta(days=1)
    else:
        start_forecast = last_real_date + pd.Timedelta(days=1)

    end_forecast = start_forecast + pd.Timedelta(days=n_days - 1)

    df_sim = forecast_engine(df, model, features, start_forecast, end_forecast)

    # =====================
    # Tables
    # =====================
    table1 = df_sim.loc[:selected_date].tail(5).copy()
    table1["Type"] = np.where(table1.index <= last_real_date, "Historical", "Forecast")
    table1["Day"] = table1.index.day_name()

    table2 = df_sim.loc[start_forecast:end_forecast].copy()
    table2["Type"] = "Forecast"
    table2["Day"] = table2.index.day_name()

    # =====================
    # Display tables
    # =====================
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 آخر 5 أيام")
        st.dataframe(table1[["Day","Cups_Count","Type"]], use_container_width=True)

    with col2:
        st.markdown("### 🔮 التنبؤ")
        st.dataframe(table2[["Day","Cups_Count","Type"]], use_container_width=True)

    # =====================
    # 📈 Chart (FIXED SIZE)
    # =====================
    st.markdown("---")
    st.markdown("### 📈 الرسم البياني")

    plot_start = last_real_date - pd.Timedelta(days=50)
    plot_df = df_sim.loc[plot_start:end_forecast]

    hist = plot_df.loc[:selected_date]
    fc = plot_df.loc[start_forecast:end_forecast]

    # 🔥 الحجم الآن يتغير فعليًا
    fig, ax = plt.subplots(figsize=(5,2))   # صغير وواضح

    ax.plot(hist.index,
            hist["Cups_Count"],
            color="#5c4033",
            linewidth=1.3,
            label="Historical")

    ax.plot(fc.index,
            fc["Cups_Count"],
            color="#d2691e",
            linestyle="--",
            linewidth=1.6,
            label="Forecast")

    ax.axvline(selected_date, color="gray", linestyle=":")

    ax.set_facecolor("#fff3e6")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45, fontsize=7)

    ax.tick_params(axis='y', labelsize=8)

    ax.legend(fontsize=8)
    ax.grid(alpha=0.15)

    st.pyplot(fig)
