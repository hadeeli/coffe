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

# 🎨 خلفية عامة بسيطة
st.markdown("""
<style>
body {
    background-color: #f7f3ef;
}
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# =====================
# عنوان رئيسي
# =====================
st.markdown("""
<div style='text-align:center; background-color:#6f4e37; padding:15px; border-radius:12px; color:white;'>
<h2>☕ نظام التنبؤ الذكي لمبيعات القهوة</h2>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

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
def forecast_engine(df, model, features, end_date):

    current = df.copy()

    dates = pd.date_range(
        df.index[-1] + pd.Timedelta(days=1),
        end_date
    )

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
# تشغيل
# =====================
if st.button("🔮 تشغيل التنبؤ"):

    forecast_start = selected_date + pd.Timedelta(days=1)
    forecast_end = selected_date + pd.Timedelta(days=n_days)

    df_sim = forecast_engine(df, model, features, forecast_end)

    # =====================
    # الجداول
    # =====================
    table1 = df_sim.loc[:selected_date].iloc[:-1].tail(5).copy()
    table2 = df_sim.loc[forecast_start:forecast_end].copy()

    # =====================
    # تنسيق الجداول
    # =====================
    table1["Type"] = np.where(table1.index <= last_real_date, "Historical", "Forecast")
    table1["Day"] = table1.index.day_name()
    table1 = table1[["Day", "Cups_Count", "Type"]]

    table2["Type"] = np.where(table2.index <= last_real_date, "Historical", "Forecast")
    table2["Day"] = table2.index.day_name()
    table2 = table2[["Day", "Cups_Count", "Type"]]

    # =====================
    # عرض الجداول جنب بعض
    # =====================
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style='background-color:#fff3e6; padding:10px; border-radius:10px;'>
        <h4 style='color:#6f4e37;'>📊 آخر 5 أيام</h4>
        </div>
        """, unsafe_allow_html=True)

        st.dataframe(table1.rename(columns={
            "Day": "اسم اليوم",
            "Cups_Count": "عدد الأكواب",
            "Type": "نوع البيانات"
        }), use_container_width=True)

    with col2:
        st.markdown("""
        <div style='background-color:#fff3e6; padding:10px; border-radius:10px;'>
        <h4 style='color:#6f4e37;'>🔮 جدول التنبؤ</h4>
        </div>
        """, unsafe_allow_html=True)

        st.dataframe(table2.rename(columns={
            "Day": "اسم اليوم",
            "Cups_Count": "عدد الأكواب",
            "Type": "نوع البيانات"
        }), use_container_width=True)

    # =====================
    # 📈 الرسم
    # =====================
    st.markdown("---")

    st.markdown("""
    <div style='background-color:#6f4e37; padding:10px; border-radius:10px; text-align:center; color:white;'>
    <h4>📈 منحنى الطلب على القهوة</h4>
    </div>
    """, unsafe_allow_html=True)

    plot_start = last_real_date - pd.Timedelta(days=50)
    plot_end = forecast_end

    plot_df = df_sim.loc[plot_start:plot_end]

    hist = plot_df.loc[:selected_date]
    fc = plot_df.loc[selected_date:]

    fig, ax = plt.subplots(figsize=(12,5))

    # 🎨 ألوان متناسقة مع القهوة
    ax.plot(hist.index, hist["Cups_Count"],
            color="#6f4e37",
            label="Historical (Coffee Trend)")

    ax.plot(fc.index, fc["Cups_Count"],
            color="#d2691e",
            linestyle="--",
            label="Forecast (Expected Demand)")

    ax.axvline(selected_date, color="gray", linestyle=":")

    ax.set_facecolor("#f7f3ef")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)

    ax.legend()
    ax.grid(alpha=0.2)

    st.pyplot(fig)
