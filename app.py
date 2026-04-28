import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# =====================
# إعداد الصفحة وتنسيق CSS
# =====================
st.set_page_config(page_title="Coffee Forecast Pro", layout="wide", initial_sidebar_state="collapsed")

# تصميم الواجهة باستخدام CSS
st.markdown("""
<style>
    /* الخلفية العامة */
    .stApp {
        background-color: #FDFBF9;
    }
    
    /* تنسيق الحاويات (Cards) */
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #EAE0D5;
        margin-bottom: 20px;
    }
    
    /* العناوين */
    h1, h2, h3 {
        color: #5E3023;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* تنسيق زر التشغيل */
    .stButton>button {
        width: 100%;
        border-radius: 25px;
        background-color: #895737;
        color: white;
        font-weight: bold;
        border: none;
        padding: 10px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #5E3023;
        color: #C19A6B;
    }
</style>
""", unsafe_allow_html=True)

# =====================
# الهيدر (Header)
# =====================
st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1 style="margin-bottom: 0;">☕ AI Coffee Forecast</h1>
        <p style="color: #895737; font-size: 1.1rem;">نظام التحليل الذكي للتنبؤ بطلب المبيعات</p>
    </div>
""", unsafe_allow_html=True)

# =====================
# تحميل الموديل والبيانات
# =====================
@st.cache_resource
def load_assets():
    model = joblib.load("xgb_model.pkl")
    features = joblib.load("features.pkl")
    return model, features

@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")
    df = df.asfreq("D").fillna(0)
    return df

model, features = load_assets()
df = load_data()
last_real_date = df.index[-1]

# =====================
# الشريط الجانبي أو لوحة التحكم
# =====================
with st.container():
    col1, col2, col3 = st.columns([1, 1, 0.8])
    with col1:
        selected_date = st.date_input("📅 تاريخ بداية التنبؤ", value=last_real_date + pd.Timedelta(days=1))
    with col2:
        n_days = st.number_input("📆 عدد الأيام", min_value=1, max_value=30, value=7)
    with col3:
        st.write(" ") # موازنة مساحة
        st.write(" ")
        run_btn = st.button("🔮 بدء التنبؤ الذكي")

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

# =====================
# عرض النتائج
# =====================
if run_btn:
    forecast_end = selected_date + pd.Timedelta(days=n_days - 1)
    df_sim = forecast_engine(df, model, features, forecast_end)

    # تجهيز الجداول
    def prep_table(data_subset):
        res = data_subset.copy()
        res["Date_Col"] = res.index.strftime("%Y-%m-%d")
        res["Day"] = res.index.day_name()
        res["Type"] = np.where(res.index <= last_real_date, "Actual", "Forecast")
        return res[["Date_Col", "Day", "Cups_Count", "Type"]]

    table1 = prep_table(df_sim.loc[:selected_date].iloc[:-1].tail(5))
    table2 = prep_table(df_sim.loc[selected_date:forecast_end])

    # عرض الجداول بتصميم أنيق
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='metric-card'><h4>📊 السجل الأخير</h4>", unsafe_allow_html=True)
        st.dataframe(table1.rename(columns={"Date_Col":"التاريخ","Day":"اليوم","Cups_Count":"الأكواب","Type":"الحالة"}), 
                     hide_index=True, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='metric-card'><h4>🔮 نتائج التنبؤ</h4>", unsafe_allow_html=True)
        st.dataframe(table2.rename(columns={"Date_Col":"التاريخ","Day":"اليوم","Cups_Count":"الأكواب","Type":"الحالة"}), 
                     hide_index=True, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # =====================
    # الرسم البياني (تصميم أصغر ومتناسق)
    # =====================
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; margin-bottom: 20px;'>📉 تحليل اتجاه الطلب المستقبلي</h4>", unsafe_allow_html=True)
    
    # تحديد نطاق الرسم (أصغر قليلاً للتركيز)
    plot_start = selected_date - pd.Timedelta(days=14)
    plot_df = df_sim.loc[plot_start:forecast_end]

    # رسم بحجم أصغر (8x3 بدلاً من 10x4)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    sns.lineplot(x=plot_df.index, y=plot_df["Cups_Count"], ax=ax, color="#EAE0D5", alpha=0.5)
    
    # تمييز الماضي والمستقبل
    hist = plot_df.loc[:selected_date]
    fc = plot_df.loc[selected_date:]
    
    ax.plot(hist.index, hist["Cups_Count"], color="#5E3023", linewidth=2.5, marker='o', markersize=4, label="Actual")
    ax.plot(fc.index, fc["Cups_Count"], color="#D4A373", linewidth=3, linestyle='--', marker='s', markersize=5, label="AI Forecast")
    
    # تظليل منطقة التنبؤ
    ax.axvspan(selected_date, forecast_end, color='#D4A373', alpha=0.1)
    
    # تنسيق المحاور
    ax.set_facecolor("white")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.xticks(rotation=0, fontsize=9)
    plt.yticks(fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("Cups", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    sns.despine() # حذف الإطارات العلوية واليمنى
    
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)
