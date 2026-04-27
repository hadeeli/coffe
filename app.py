import streamlit as st
import pandas as pd
import joblib

# 1. تحميل الموديل
model = joblib.load('coffee_model.pkl')

st.title("☕ نظام التنبؤ الذكي للمقهى")

# 2. إدخال القيم (يجب توفير القيم التي يحتاجها الموديل)
# ملاحظة: في المشروع الحقيقي، هذه القيم تُسحب من قاعدة البيانات، 
# لكن للواجهة الآن سنقوم بإدخالها يدوياً لتجربة الموديل
col1, col2 = st.columns(2)

with col1:
    lag_1 = st.number_input("مبيعات أمس (lag_1)", min_value=0.0)
    ma_7 = st.number_input("متوسط مبيعات آخر 7 أيام (MA_7)", min_value=0.0)
    month = st.slider("الشهر", 1, 12, 5)

with col2:
    lag_7 = st.number_input("مبيعات نفس اليوم الأسبوع الماضي (lag_7)", min_value=0.0)
    day_of_week = st.selectbox("اليوم", [0,1,2,3,4,5,6], format_func=lambda x: ["الأثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت", "الأحد"][x])
    is_weekend = 1 if day_of_week in [4, 5] else 0

# 3. زر التوقع
if st.button("توقع الآن 🚀"):
    # تجهيز الجدول بالـ 10 أعمدة المطلوبة بنفس الترتيب تماماً
    input_data = pd.DataFrame([[
        ma_7,          # MA_7
        20.0,          # MA_30 (قيمة افتراضية مؤقتة)
        day_of_week,   # day_of_week
        month,         # month
        is_weekend,    # is_weekend
        lag_1,         # lag_1
        lag_7,         # lag_7
        ma_7,          # rolling_mean_7 (غالباً نفس MA_7)
        2.0,           # rolling_std_7 (قيمة افتراضية)
        0.0            # seasonal_diff (قيمة افتراضية)
    ]], columns=['MA_7', 'MA_30', 'day_of_week', 'month', 'is_weekend', 'lag_1', 'lag_7', 'rolling_mean_7', 'rolling_std_7', 'seasonal_diff'])

    # تنفيذ التوقع
    prediction = model.predict(input_data)
    
    st.success(f"النتيجة المتوقعة: {int(prediction[0])} كوب")
