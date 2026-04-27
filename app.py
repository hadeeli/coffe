import streamlit as st
import pandas as pd
import joblib
import datetime
import numpy as np
from scipy.special import inv_boxcox
from scipy import stats # نحتاجها للتحويل

# 1. تحميل الموديل وقيمة Lambda
model = joblib.load('coffee_model.pkl')
lmbda = 0.49 

# 2. تحميل البيانات
@st.cache_data
def load_historical_data():
    df = pd.read_csv('coffee_data.csv') 
    return df

df_history = load_historical_data()

# --- دالة مساعدة لتحويل القيم قبل إرسالها للموديل ---
def transform_value(val):
    # تحويل Box-Cox يدوياً لضمان التطابق
    return (np.power(val + 1, lmbda) - 1) / lmbda

if df_history is not None:
    # نأخذ آخر قيم حقيقية من الملف
    last_actual = df_history['cups_sold'].iloc[-1]
    avg_7_actual = df_history['cups_sold'].tail(7).mean()
    avg_30_actual = df_history['cups_sold'].tail(30).mean()
    std_7_actual = df_history['cups_sold'].tail(7).std()
    
    # تحويل هذه القيم لتناسب "لغة" الموديل
    lag_1_tr = transform_value(last_actual)
    ma_7_tr = transform_value(avg_7_actual)
    ma_30_tr = transform_value(avg_30_actual)
    std_7_tr = transform_value(std_7_actual) if std_7_actual > 0 else 0
else:
    # قيم افتراضية في حال فشل التحميل
    lag_1_tr, ma_7_tr, ma_30_tr, std_7_tr = 2.5, 2.5, 2.5, 0.5

# --- واجهة المستخدم ---
st.title("☕ متنبئ مبيعات القهوة")
selected_date = st.date_input("اختر التاريخ:", datetime.date.today() + datetime.timedelta(days=1))

if st.button("توقع المبيعات 🚀"):
    # تجهيز الـ 10 ميزات
    day_of_week = selected_date.weekday()
    month = selected_date.month
    is_weekend = 1 if day_of_week in [4, 5] else 0

    input_data = pd.DataFrame([[
        ma_7_tr,        # MA_7
        ma_30_tr,       # MA_30
        day_of_week,    # day_of_week (لا يحتاج تحويل)
        month,          # month (لا يحتاج تحويل)
        is_weekend,     # is_weekend (لا يحتاج تحويل)
        lag_1_tr,       # lag_1
        lag_1_tr,       # lag_7 (افتراضياً)
        ma_7_tr,        # rolling_mean_7
        std_7_tr,       # rolling_std_7
        0.0             # seasonal_diff
    ]], columns=['MA_7', 'MA_30', 'day_of_week', 'month', 'is_weekend', 'lag_1', 'lag_7', 'rolling_mean_7', 'rolling_std_7', 'seasonal_diff'])

    # التوقع
    prediction_tr = model.predict(input_data)[0]
    
    # عكس التحويل
    prediction_actual = inv_boxcox(prediction_tr, lmbda) - 1
    
    # النتيجة النهائية
    final_result = max(0, int(round(prediction_actual)))
    
    st.metric(label="الكمية المتوقعة", value=f"{final_result} كوب")
