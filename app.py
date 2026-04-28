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

st.markdown("""
<style>
body {
    background-color: #f5f1ec;
    font-family: Arial;
}
table {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# =====================
# العنوان
# =====================
st.markdown("""
<div style="
    background-color:#6f4e37;
    padding:15px;
    border-radius:12px;
    text-align:center;
    color:white;">
    <h2>☕ نظام التنبؤ الذكي للقهوة</h2>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# =====================
# تحميل النموذج والميزات
# =====================
# ملاحظة: تأكدي أن الملفات موجودة في نفس المجلد
model = joblib.load("xgb_model.pkl")
features = joblib.load("features.pkl")

# =====================
# تحميل وتحضير البيانات
# =====================
df = pd.read_csv("data.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")
df.set_index("Date", inplace=True)
df = df.asfreq("D").fillna(0)

last_real_date = df.index[-1]

# =====================
# مدخلات المستخدم (Inputs)
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
# محرك التنبؤ (Forecast Engine)
# =====================
def forecast_engine(df, model, features, end_date):
    current = df.copy()
    
    # تحديد نطاق التواريخ المطلوبة للتنبؤ
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
        row["lag_7"] = current["Cups_Count"].iloc[-7] if len(current) >= 7 else current["Cups_Count"].iloc[-1]

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
# تشغيل التنبؤ عند الضغط على الزر
# =====================
if st.button("🔮 تشغيل التنبؤ"):

    # حساب تاريخ النهاية بناءً على اختيار المستخدم
    forecast_end = selected_date + pd.Timedelta(days=n_days - 1)

    # تشغيل المحرك
    df_sim = forecast_engine(df, model, features, forecast_end)

    # ---------------------
    # 📊 تجهيز الجدول 1 (آخر 5 أيام)
    # ---------------------
    table1 = df_sim.loc[:selected_date].iloc[:-1].tail(5).copy()
    table1["Type"] = np.where(table1.index <= last_real_date, "Historical", "Forecast")
    table1["Day"] = table1.index.day_name()
    
    # ✅ حذف الوقت: تحويل التاريخ لنص بدون أصفار الوقت
    table1["Date_Col"] = table1.index.strftime("%Y-%m-%d")
    table1 = table1[["Date_Col", "Day", "Cups_Count", "Type"]]

    # ---------------------
    # 📊 تجهيز الجدول 2 (جدول التنبؤ)
    # ---------------------
    table2 = df_sim.loc[selected_date:forecast_end].copy()
    table2["Type"] = np.where(table2.index <= last_real_date, "Historical", "Forecast")
    table2["Day"] = table2.index.day_name()

    # ✅ حذف الوقت: تحويل التاريخ لنص بدون أصفار الوقت
    table2["Date_Col"] = table2.index.strftime("%Y-%m-%d")
    table2 = table2[["Date_Col", "Day", "Cups_Count", "Type"]]

    # =====================
    # عرض الجداول في أعمدة
    # =====================
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div style='background:#fff3e6; padding:10px; border-radius:10px; margin-bottom:10px;'>
        <h4 style='color:#6f4e37; margin:0;'>📊 آخر 5 أيام</h4>
        </div>
        """, unsafe_allow_html=True)

        st.dataframe(
            table1.rename(columns={
                "Date_Col": "التاريخ",
                "Day": "اسم اليوم",
                "Cups_Count": "عدد الأكواب",
                "Type": "نوع البيانات"
            }),
            use_container_width=True,
            hide_index=True # إخفاء الفهرس الجانبي
        )

    with c2:
        st.markdown("""
        <div style='background:#fff3e6; padding:10px; border-radius:10px; margin-bottom:10px;'>
        <h4 style='color:#6f4e37; margin:0;'>🔮 جدول التنبؤ</h4>
        </div>
        """, unsafe_allow_html=True)

        st.dataframe(
            table2.rename(columns={
                "Date_Col": "التاريخ",
                "Day": "اسم اليوم",
                "Cups_Count": "عدد الأكواب",
                "Type": "نوع البيانات"
            }),
            use_container_width=True,
            hide_index=True # إخفاء الفهرس الجانبي
        )

    # =====================
    # 📈 الرسم البياني
    # =====================
    st.markdown("---")
    st.markdown("<h3 style='text-align:center; color:#6f4e37;'>📈 منحنى الطلب على القهوة</h3>", unsafe_allow_html=True)

    # نحدد نطاق الرسم ليظهر آخر 50 يوم + التنبؤ
    plot_start = last_real_date - pd.Timedelta(days=50)
    plot_df = df_sim.loc[plot_start:forecast_end]

    hist = plot_df.loc[:selected_date]
    fc = plot_df.loc[selected_date:]

    fig, ax = plt.subplots(figsize=(10, 4))
    
    # رسم البيانات التاريخية والتوقعات
    ax.plot(hist.index, hist["Cups_Count"], color="#6f4e37", linewidth=2, label="Historical")
    ax.plot(fc.index, fc["Cups_Count"], color="#d2691e", linestyle="--", linewidth=3, label="Forecast")

    # خط عمودي يفصل بين الماضي والمستقبل
    ax.axvline(selected_date, color="gray", linestyle=":", alpha=0.5)

    # تحسين مظهر الرسم
    ax.set_facecolor("#f5f1ec")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)
    ax.set_ylabel("Number of Cups")
    ax.legend()
    ax.grid(alpha=0.2)

    st.pyplot(fig)
