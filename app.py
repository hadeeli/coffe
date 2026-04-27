import streamlit as st
import pandas as pd
import joblib
import datetime

# 1. تحميل الموديل
model = joblib.load('coffee_model.pkl')

# 2. تحميل آخر بيانات المبيعات (نحتاجها لحساب المتوسطات والـ Lag)
# افترضي أن عندك ملف اسمه coffee_data.csv فيه عمود المبيعات التاريخية
@st.cache_data
def load_historical_data():
    # هنا نقرأ الملف الذي رفعتيه مع الموديل على GitHub
    df = pd.read_csv('coffee_data.csv') 
    return df

try:
    df_history = load_historical_data()
    last_sales = df_history['cups_sold'].iloc[-1] # آخر مبيعات مسجلة (lag_1)
    avg_7 = df_history['cups_sold'].tail(7).mean() # متوسط آخر 7 أيام (MA_7)
    avg_30 = df_history['cups_sold'].tail(30).mean() # متوسط آخر 30 يوم (MA_30)
except:
    # قيم افتراضية في حال عدم وجود ملف بيانات
    last_sales, avg_7, avg_30 = 25.0, 22.0, 20.0

st.title("☕ متنبئ مبيعات القهوة الذكي")
st.write("اختر التاريخ وسأخبرك بالكمية المتوقعة!")

# 3. المدخل الوحيد للمستخدم
selected_date = st.date_input("اختر اليوم:", datetime.date.today() + datetime.timedelta(days=1))

# 4. الحسابات التلقائية (خلف الكواليس)
day_of_week = selected_date.weekday()
month = selected_date.month
is_weekend = 1 if day_of_week in [4, 5] else 0

if st.button("توقع المبيعات 🚀"):
    # تجهيز الـ 10 أعمدة المطلوبة للموديل تلقائياً
    input_data = pd.DataFrame([[
        avg_7,         # MA_7 (محسوبة)
        avg_30,        # MA_30 (محسوبة)
        day_of_week,   # من التاريخ
        month,         # من التاريخ
        is_weekend,    # من التاريخ
        last_sales,    # lag_1 (من آخر يوم في الملف)
        last_sales,    # lag_7 (افتراضياً نضعها كآخر مبيعات)
        avg_7,         # rolling_mean_7
        2.5,           # rolling_std_7 (قيمة ثابتة)
        0.0            # seasonal_diff
    ]], columns=['MA_7', 'MA_30', 'day_of_week', 'month', 'is_weekend', 'lag_1', 'lag_7', 'rolling_mean_7', 'rolling_std_7', 'seasonal_diff'])

    prediction = model.predict(input_data)
    
    # 5. عرض النتيجة النهائية
    st.balloons()
    st.metric(label="التوقعات لهذا اليوم", value=f"{int(prediction[0])} كوب")
    
    if prediction[0] > avg_7:
        st.warning("⚠️ يتوقع زحام أكثر من المعتاد، جهزي المزيد من البن!")
    else:
        st.info("☕ يوم هادئ ومناسب للتنظيم.")
