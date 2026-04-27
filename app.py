import streamlit as st
import pandas as pd
import joblib
import datetime
import numpy as np
from scipy.special import inv_boxcox  # استيراد الدالة العكسية

# 1. تحميل الموديل وتحديد قيمة Lambda
model = joblib.load('coffee_model.pkl')
lmbda = 0.49  # القيمة التي استخرجتيها من النوت بوك

# 2. تحميل آخر بيانات المبيعات
@st.cache_data
def load_historical_data():
    try:
        df = pd.read_csv('coffee_data.csv') 
        return df
    except:
        return None

df_history = load_historical_data()

if df_history is not None:
    # حساب القيم من الملف المرفوع
    last_sales_actual = df_history['cups_sold'].iloc[-1]
    # ملاحظة: الموديل تدرب على بيانات محولة، لذا نحتاج لتحويل القيم المدخلة أيضاً
    # لكن للتبسيط، سنستخدم المتوسطات الحسابية كتقريب جيد
    last_sales_transformed = (df_history['cups_sold'].iloc[-1]**lmbda - 1) / lmbda
    avg_7_transformed = ((df_history['cups_sold'].tail(7).mean())**lmbda - 1) / lmbda
    avg_30_transformed = ((df_history['cups_sold'].tail(30).mean())**lmbda - 1) / lmbda
    current_avg = df_history['cups_sold'].tail(7).mean()
else:
    # قيم افتراضية محولة تقريبياً في حال تعذر التحميل
    last_sales_transformed, avg_7_transformed, avg_30_transformed = 2.5, 2.3, 2.2
    current_avg = 25.0

st.title("☕ متنبئ مبيعات القهوة الذكي")
st.write("اختر التاريخ وسأخبرك بالكمية المتوقعة!")

# 3. مدخلات المستخدم
selected_date = st.date_input("اختر اليوم:", datetime.date.today() + datetime.timedelta(days=1))

# 4. الحسابات التلقائية
day_of_week = selected_date.weekday()
month = selected_date.month
is_weekend = 1 if day_of_week in [4, 5] else 0

if st.button("توقع المبيعات 🚀"):
    # تجهيز الـ 10 أعمدة المطلوبة (بصيغتها المحولة)
    input_data = pd.DataFrame([[
        avg_7_transformed,      # MA_7
        avg_30_transformed,     # MA_30
        day_of_week,            # day_of_week
        month,                  # month
        is_weekend,             # is_weekend
        last_sales_transformed, # lag_1
        last_sales_transformed, # lag_7
        avg_7_transformed,      # rolling_mean_7
        0.5,                    # rolling_std_7 (قيمة ثابتة تقريبية)
        0.0                     # seasonal_diff
    ]], columns=['MA_7', 'MA_30', 'day_of_week', 'month', 'is_weekend', 'lag_1', 'lag_7', 'rolling_mean_7', 'rolling_std_7', 'seasonal_diff'])

    # التوقع (النتيجة ستكون بصيغة Box-Cox)
    prediction_transformed = model.predict(input_data)[0]
    
    # 5. عكس التحويل للحصول على عدد الأكواب الحقيقي
    # المعادلة: (pred * lmbda + 1) ^ (1/lmbda) - 1
    prediction_actual = inv_boxcox(prediction_transformed, lmbda) - 1
    
    # ضمان عدم ظهور أرقام سالبة
    final_result = max(0, int(round(prediction_actual)))

    # عرض النتيجة النهائية
    st.balloons()
    st.metric(label="التوقعات لهذا اليوم", value=f"{final_result} كوب")
    
    if final_result > current_avg:
        st.warning(f"⚠️ يتوقع زحام أعلى من المتوسط ({int(current_avg)})، جهزي المزيد من البن!")
    else:
        st.info(f"☕ يوم طبيعي، المتوسط المعتاد هو {int(current_avg)} كوب.")
