import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# =====================
# إعداد الصفحة والخلفية
# =====================
st.set_page_config(page_title="Coffee Forecast", layout="wide")

# إضافة خلفية ملونة للصفحة بالكامل وتحسين الخطوط
st.markdown("""
<style>
    /* خلفية التطبيق كاملة */
    .stApp {
        background-color: #fcfaf8; 
    }
    /* تحسين الخطوط */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# =====================
# العنوان الرئيسي
# =====================
st.markdown("""
<div style="
    background-color:#6f4e37;
    padding:15px;
    border-radius:12px;
    text-align:center;
    color:white;
    margin-bottom: 25px;">
    <h1 style="margin:0; font-size: 26px;">☕ نظام التنبؤ الذكي للقهوة</h1>
</div>
""", unsafe_allow_html=True)

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
# مدخلات المستخدم
# =====================
col1, col2 = st.columns(2)
with col1:
    selected_date = st.date_input("📅 تاريخ بداية التنبؤ")
with col2:
    n_days = st.number_input("📆 عدد أيام التنبؤ", min_value=1, max_value=30, value=5, step=1)

selected_date = pd.to_datetime(selected_date)

# =====================
# محرك التنبؤ
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
# تنفيذ التنبؤ والعرض
# =====================
if st.button("🔮 تشغيل التنبؤ"):
    forecast_end = selected_date + pd.Timedelta(days=n_days - 1)
    df_sim = forecast_engine(df, model, features, forecast_end)

    # قاموس تحويل أيام الأسبوع للعربية
    days_ar = {
        'Monday': 'الاثنين',
        'Tuesday': 'الثلاثاء',
        'Wednesday': 'الأربعاء',
        'Thursday': 'الخميس',
        'Friday': 'الجمعة',
        'Saturday': 'السبت',
        'Sunday': 'الأحد'
    }

    # تجهيز الجداول بأسماء احترافية
    def clean_table(data_subset):
        res = data_subset.copy()
        res["التاريخ"] = res.index.strftime("%Y-%m-%d")
        # تحويل اسم اليوم للعربية باستخدام القاموس
        res["اليوم"] = res.index.day_name().map(days_ar)
        res["الحالة"] = np.where(res.index <= last_real_date, "Actual", "Forecast")
        return res[["التاريخ", "اليوم", "Cups_Count", "الحالة"]]

    t1 = clean_table(df_sim.loc[:selected_date].iloc[:-1].tail(5))
    t2 = clean_table(df_sim.loc[selected_date:forecast_end])

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<h4 style='color:#6f4e37;'>📊 سجل المبيعات الأخيرة</h4>", unsafe_allow_html=True)
        st.dataframe(t1.rename(columns={"Cups_Count":"الكمية (كوب)"}), use_container_width=True, hide_index=True)
    with c2:
        st.markdown("<h4 style='color:#6f4e37;'>🔮 التوقعات الذكية للطلب</h4>", unsafe_allow_html=True)
        st.dataframe(t2.rename(columns={"Cups_Count":"الكمية (كوب)"}), use_container_width=True, hide_index=True)

    # =====================
    # الرسم البياني المطور
    # =====================
    st.markdown("---")
    st.markdown("<h4 style='text-align:center; color:#6f4e37;'>📈 الرسم البياني للتنبؤ</h4>", unsafe_allow_html=True)

    plot_start = selected_date - pd.Timedelta(days=14)
    plot_df = df_sim.loc[plot_start:forecast_end]

    fig, ax = plt.subplots(figsize=(9, 4))
    
    hist = plot_df.loc[:selected_date]
    fc = plot_df.loc[selected_date:]

    ax.plot(hist.index, hist["Cups_Count"], color="#6f4e37", linewidth=1.5, marker='o', markersize=3, label="Actual")
    ax.plot(fc.index, fc["Cups_Count"], color="#d2691e", linestyle="--", linewidth=1.8, marker='s', markersize=4, label="Forecast")

    ax.set_facecolor("white")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.xticks(rotation=0, fontsize=9)
    plt.yticks(fontsize=9)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(axis='y', alpha=0.3)

    for i, v in enumerate(fc["Cups_Count"]):
        ax.text(fc.index[i], v + 0.5, str(int(v)), color='#d2691e', fontsize=8, ha='center', fontweight='bold')

    left, mid, right = st.columns([0.1, 0.8, 0.1])
    with mid:
        st.pyplot(fig)
