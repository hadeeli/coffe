import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# =====================
# إعداد الصفحة وتنسيق الخطوط
# =====================
st.set_page_config(page_title="Coffee Forecast", layout="wide")

st.markdown("""
<style>
    /* تحسين الخطوط في التطبيق كامل */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main {
        background-color: #f5f1ec;
    }
    /* تكبير خط العناوين داخل الجداول */
    .stDataFrame {
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# =====================
# العنوان (التنسيق البسيط الذي أعجبك)
# =====================
st.markdown("""
<div style="
    background-color:#6f4e37;
    padding:15px;
    border-radius:12px;
    text-align:center;
    color:white;">
    <h1 style="margin:0; font-size: 28px;">☕ نظام التنبؤ الذكي للقهوة</h1>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# =====================
# تحميل النموذج والبيانات
# =====================
model = joblib.load("xgb_model.pkl")
features = joblib.load("features.pkl")

df = pd.read_csv("data.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").set_index("Date")
df = df.asfreq("D").fillna(0)
last_real_date = df.index[-1]

# =====================
# المدخلات (Inputs)
# =====================
col1, col2 = st.columns(2)
with col1:
    selected_date = st.date_input("📅 اختر تاريخ بداية التنبؤ")
with col2:
    n_days = st.number_input("📆 عدد أيام التنبؤ", min_value=1, max_value=30, value=5, step=1, format="%d")

selected_date = pd.to_datetime(selected_date)

# =====================
# محرك التنبؤ (Engine)
# =====================
def forecast_engine(df, model, features, end_date):
    current = df.copy()
    dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), end_date)
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
        current = pd.concat([current, pd.DataFrame({"Cups_Count": pred}, index=[d])])
    return current

st.markdown("---")

# =====================
# تشغيل وعرض النتائج
# =====================
if st.button("🔮 تشغيل التنبؤ"):
    forecast_end = selected_date + pd.Timedelta(days=n_days - 1)
    df_sim = forecast_engine(df, model, features, forecast_end)

    # تجهيز الجداول
    def clean_table(data_subset):
        res = data_subset.copy()
        res["التاريخ"] = res.index.strftime("%Y-%m-%d")
        res["اسم اليوم"] = res.index.day_name()
        res["نوع البيانات"] = np.where(res.index <= last_real_date, "Historical", "Forecast")
        return res[["التاريخ", "اسم اليوم", "Cups_Count", "نوع البيانات"]]

    t1 = clean_table(df_sim.loc[:selected_date].iloc[:-1].tail(5))
    t2 = clean_table(df_sim.loc[selected_date:forecast_end])

    # عرض الجداول
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<h4 style='color:#6f4e37;'>📊 آخر 5 أيام</h4>", unsafe_allow_html=True)
        st.dataframe(t1.rename(columns={"Cups_Count":"عدد الأكواب"}), use_container_width=True, hide_index=True)
    with c2:
        st.markdown("<h4 style='color:#6f4e37;'>🔮 جدول التنبؤ</h4>", unsafe_allow_html=True)
        st.dataframe(t2.rename(columns={"Cups_Count":"عدد الأكواب"}), use_container_width=True, hide_index=True)

    # =====================
    # الرسم البياني (حجم أصغر وأكثر تناسقاً)
    # =====================
    st.markdown("---")
    st.markdown("<h4 style='text-align:center; color:#6f4e37;'>📈 منحنى الطلب المتوقع</h4>", unsafe_allow_html=True)

    # تصغير نطاق الرسم ليصبح أكثر تركيزاً
    plot_start = selected_date - pd.Timedelta(days=10)
    plot_df = df_sim.loc[plot_start:forecast_end]

    # تصغير حجم الرسم (7x3) ليكون متناسقاً جداً
    fig, ax = plt.subplots(figsize=(7, 3))
    
    # تقسيم البيانات للرسم
    hist = plot_df.loc[:selected_date]
    fc = plot_df.loc[selected_date:]

    # الرسم الفعلي
    ax.plot(hist.index, hist["Cups_Count"], color="#6f4e37", linewidth=2, marker='o', markersize=4, label="Historical")
    ax.plot(fc.index, fc["Cups_Count"], color="#d2691e", linestyle="--", linewidth=2.5, marker='s', markersize=5, label="Forecast")

    # تنسيق المحاور والخطوط
    ax.set_facecolor("#fcfaf8")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.xticks(rotation=0, fontsize=8)
    plt.yticks(fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    # إضافة أرقام التنبؤ بشكل صغير
    for i, v in enumerate(fc["Cups_Count"]):
        ax.text(fc.index[i], v + 0.5, str(int(v)), color='#d2691e', fontweight='bold', fontsize=8, ha='center')

    # عرض الرسم في Streamlit مع التحكم بالعرض
    st.columns([1, 2, 1])[1].pyplot(fig) # وضع الرسم في العمود الأوسط ليظهر في المنتصف وبحجم متناسق
