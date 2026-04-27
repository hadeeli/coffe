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

st.markdown("<h1 style='text-align:center;'>☕ نظام التنبؤ الذكي للقهوة</h1>", unsafe_allow_html=True)

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
# UI Inputs
# =====================
selected_date = st.date_input("📅 اختر التاريخ")
selected_date = pd.to_datetime(selected_date)

n_future = st.slider("📆 عدد أيام التنبؤ بعد التاريخ المختار", 1, 14, 7)

st.markdown("---")

# =====================
# هل التاريخ موجود؟
# =====================
date_exists = selected_date in df.index

st.info(f"📌 التاريخ موجود في البيانات: {date_exists}")

# =====================
# تجهيز البداية
# =====================
df_work = df.copy()
last_real_date = df.index[-1]

# =====================
# تحديد نقطة البداية
# =====================
if date_exists:

    # نوقف عند التاريخ المختار
    df_work = df.loc[:selected_date].copy()
    forecast_start = selected_date

else:

    # نستخدم كامل البيانات
    forecast_start = last_real_date

# =====================
# عرض آخر 5 أيام (real + predicted لاحقًا)
# =====================
df_work["day_name"] = df_work.index.day_name()

st.subheader("📊 آخر 5 أيام")
st.dataframe(
    df_work.tail(5)[["day_name", "Cups_Count"]]
    .rename(columns={"Cups_Count": "عدد الأكواب"})
)

# =====================
# زر التنفيذ
# =====================
if st.button("🔮 تشغيل التنبؤ"):

    df_future = df.copy()
    future_values = []
    future_dates = []

    # إذا التاريخ غير موجود → نحسب حتى الوصول له + بعده
    if not date_exists:
        total_days = (selected_date - last_real_date).days + n_future
    else:
        total_days = n_future

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

        df_future = pd.concat([df_future, pd.DataFrame({"Cups_Count": pred}, index=[next_date])])

        future_values.append(pred)
        future_dates.append(next_date)

    # =====================
    # تحديد جزء العرض (آخر 5 أيام + forecast)
    # =====================
    full_series = df_future.copy()

    full_series["day_name"] = full_series.index.day_name()

    display_df = full_series.tail(5 + n_future)

    st.subheader("📅 النتائج (5 أيام + التنبؤ)")

    st.dataframe(
        display_df[["day_name", "Cups_Count"]]
        .rename(columns={
            "day_name": "اسم اليوم",
            "Cups_Count": "عدد الأكواب"
        })
    )

    # =====================
    # Forecast table
    # =====================
    forecast_df = pd.DataFrame({
        "التاريخ": future_dates[-n_future:],
        "اسم اليوم": [d.day_name() for d in future_dates[-n_future:]],
        "عدد الأكواب": future_values[-n_future:]
    })

    st.subheader("📊 جدول التنبؤ فقط")
    st.dataframe(forecast_df)

    # =====================
    # الرسم
    # =====================
    st.subheader("📈 الرسم البياني")

    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(df.index[-30:], df["Cups_Count"].iloc[-30:], label="القيم الفعلية")

    last_value = df["Cups_Count"].iloc[-1]

    all_dates = [last_real_date] + future_dates
    all_values = [last_value] + future_values

    ax.plot(all_dates, all_values,
            marker="o",
            linestyle="--",
            label="التنبؤ")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)

    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)
