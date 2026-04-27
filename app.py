import streamlit as st
import pandas as pd
import joblib
import datetime
import numpy as np
from scipy.special import inv_boxcox

# 1. تحميل الموديل وقيمة Lambda
model = joblib.load('coffee_model.pkl')
lmbda = 0.49 

# 2. تحميل البيانات
@st.cache_data
def load_historical_data():
    # قراءة الملف مع اعتبار أول عمود (التاريخ) هو الـ Index
    df = pd.read_csv('coffee_data.csv', index_col=0) 
    return df

try:
    df_history = load_historical_data()
    
    # استخدام المسميات الصحيحة من قائمتك
    # نأخذ آخر قيم حقيقية من الملف لحساب الميزات
    last_actual = df_history['Cups_Count'].iloc[-1]
    avg_7_actual = df_history['Cups_Count'].tail(7).mean()
    avg_30_actual = df_history['Cups_Count'].tail(30).mean()
    std_7_actual = df_history['Cups_Count'].tail(7).std()

    # تحويل القيم ليفهمها الموديل (لأن الموديل تدرب على Cups_Count محول)
    def transform(val):
        return (np.power(val + 1, lmbda) - 1) / lmbda

    lag_1_tr = transform(last_actual)
    ma_7_tr = transform(avg_7_actual)
    ma_30_tr = transform(avg_30_actual)
    std_7_tr = transform(std_7_actual) if std_7_actual > 0 else 0

except Exception as e:
    st.error(f"خطأ في قراءة البيانات: {e}")
    lag_1_tr, ma_7_tr, ma_30_tr, std_7_tr = 2.5, 2.5, 2.5, 0.5

# --- واجهة المستخدم ---
st.title("☕ متنبئ مبيعات القهوة الذكي")
st.write("بناءً على آخر البيانات المسجلة، إليك توقعاتنا:")

selected_date = st.date_input("اختر التاريخ المطلوب توقعه:", datetime.date.today() + datetime.timedelta(days=1))

if st.button("توقع المبيعات 🚀"):
    day_of_week = selected_date.weekday()
    month = selected_date.month
    is_weekend = 1 if day_of_week in [4, 5] else 0

    # بناء الميزات بالترتيب الصحيح الذي يطلبه الموديل
    input_data = pd.DataFrame([[
        ma_7_tr,        # MA_7
        ma_30_tr,       # MA_30
        day_of_week,    # day_of_week
        month,          # month
        is_weekend,     # is_weekend
        lag_1_tr,       # lag_1
        lag_1_tr,       # lag_7 (افتراضياً نستخدم lag_1)
        ma_7_tr,        # rolling_mean_7
        std_7_tr,       # rolling_std_7
        0.0             # seasonal_diff
    ]], columns=['MA_7', 'MA_30', 'day_of_week', 'month', 'is_weekend', 'lag_1', 'lag_7', 'rolling_mean_7', 'rolling_std_7', 'seasonal_diff'])

    # التوقع بعيداً عن الأرقام الثابتة
    prediction_tr = model.predict(input_data)[0]
    
    # عكس التحويل للحصول على عدد الأكواب
    prediction_actual = inv_boxcox(prediction_tr, lmbda) - 1
    final_result = max(0, int(round(prediction_actual)))
    
    # عرض النتيجة
    st.balloons()
    st.metric(label="الكمية المتوقعة", value=f"{final_result} كوب")
    
    st.info(f"ملاحظة: التوقع يعتمد على آخر مبيعات مسجلة في النظام وهي ({int(df_history['Cups_Count'].iloc[-1])}) كوب.")
