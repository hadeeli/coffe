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

st.markdown("<h1 style='text-align: center;'>☕ نظام التنبؤ باستهلاك القهوة</h1>", unsafe_allow_html=True)

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
# خيارات المستخدم
# =====================
mode = st.radio(
    "🎯 اختر نوع التنبؤ",
    ["🔮 تنبؤ مستقبلي", "📊 تنبؤ من تاريخ محدد"]
)

selected_date = st.date_input("📅 اختر التاريخ")
selected_date = pd.to_datetime(selected_date)

n_days = st.slider("📆 عدد أيام التنبؤ", 1, 14, 7)

st.markdown("---")

# =====================
# تحديد نقطة البداية
# =====================
if mode == "🔮 تنبؤ مستقبلي":
    df_input = df.copy()
    start_index = df.index[-1]

else:
    df_input = df.loc[:selected_date].copy()
    start_index = df_input.index[-1]

# =====================
# عرض آخر 5 أيام
# =====================
df_input["اسم اليوم"] = df_input.index.day_name()

st.subheader("📊 آخر 5 أيام")
st.dataframe(
    df_input.tail(5)[["اسم اليوم", "Cups_Count"]]
    .rename(columns={"Cups_Count": "عدد الأكواب"})
)

# =====================
# زر التنبؤ
# =====================
if st.button("🔮 تنفيذ التنبؤ"):

    future_predictions = []
    df_future = df_input.copy()

    # =====================
    # Forecast loop
    # =====================
    for i in range(n_days):

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
    # جدول النتائج
    # =====================
    future_dates = pd.date_range(
        start=start_index + pd.Timedelta(days=1),
        periods=n_days
    )

    forecast_df = pd.DataFrame({
        "التاريخ": future_dates,
        "اسم اليوم": [d.day_name() for d in future_dates],
        "عدد الأكواب": future_predictions
    })

    st.subheader("📅 جدول التنبؤ")
    st.dataframe(forecast_df)

    # =====================
    # الرسم
    # =====================
    st.subheader("📈 الرسم البياني")

    fig, ax = plt.subplots(figsize=(12,5))

    # بيانات فعلية
    ax.plot(df_input.index[-30:],
            df_input['Cups_Count'].iloc[-30:],
            label="القيم الفعلية")

    # ربط آخر نقطة مع التنبؤ
    last_date = df_input.index[-1]
    last_value = df_input['Cups_Count'].iloc[-1]

    all_dates = [last_date] + list(forecast_df["التاريخ"])
    all_values = [last_value] + list(forecast_df["عدد الأكواب"])

    ax.plot(all_dates, all_values,
            marker='o',
            linestyle='--',
            label="التنبؤ")

    # تحسين التاريخ
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)

    ax.set_title("توقع استهلاك القهوة")
    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)
