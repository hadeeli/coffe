import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# =====================
# إعداد الصفحة
# =====================
st.set_page_config(
    page_title="توقع استهلاك القهوة",
    layout="wide"
)

st.markdown("<h1 style='text-align: center;'>☕ نظام التنبؤ باستهلاك القهوة</h1>", unsafe_allow_html=True)

st.markdown("---")

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
# تجهيز بيانات قبل التاريخ
# =====================
df_filtered = df.loc[:selected_date].copy()

# عمود اسم اليوم
df_filtered["اسم اليوم"] = df_filtered.index.day_name()

st.subheader("📊 آخر 5 أيام قبل التاريخ المختار")

st.dataframe(
    df_filtered.tail(5)[["اسم اليوم", "Cups_Count"]]
    .rename(columns={
        "Cups_Count": "عدد الأكواب"
    })
)

# =====================
# زر التنبؤ
# =====================
if st.button("🔮 تنفيذ التنبؤ"):

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
    # جدول التنبؤ
    # =====================
    future_dates = pd.date_range(
        start=df_filtered.index[-1] + pd.Timedelta(days=1),
        periods=7
    )

    forecast_df = pd.DataFrame({
        "التاريخ": future_dates,
        "اسم اليوم": [d.day_name() for d in future_dates],
        "عدد الأكواب": future_predictions
    })

    st.subheader("📅 التنبؤ للأيام القادمة")
    st.dataframe(forecast_df)

    # =====================
    # الرسم
    # =====================
    st.subheader("📈 الرسم البياني للتنبؤ")

    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(df_filtered.index[-30:],
            df_filtered['Cups_Count'].iloc[-30:],
            label="القيم الفعلية")

    last_date = df_filtered.index[-1]
    last_value = df_filtered['Cups_Count'].iloc[-1]

    all_dates = [last_date] + list(forecast_df["التاريخ"])
    all_values = [last_value] + list(forecast_df["عدد الأكواب"])

    ax.plot(all_dates, all_values,
            marker='o',
            linestyle='--',
            label="التنبؤ")

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)

    ax.set_title("توقع استهلاك القهوة للأيام القادمة")
    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)
