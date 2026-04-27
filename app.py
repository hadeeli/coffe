import streamlit as st
import joblib
import pandas as pd

# 1. تحميل الموديل
model = joblib.load('coffee_model.pkl')

st.title("☕ نظام التنبؤ بمبيعات المقهى")

# 2. مدخلات المستخدم
date = st.date_input("اختر اليوم للتنبوء بمبيعاته")
prev_sales = st.number_input("مبيعات اليوم السابق", min_value=0)

# 3. معالجة المدخلات (تحويل التاريخ لميزات يفهمها الموديل)
weekday = date.weekday()
is_weekend = 1 if weekday in [4, 5] else 0 # الجمعة والسبت

if st.button("توقع المبيعات"):
    # ترتيب المدخلات بنفس ترتيب التدريب (مثال: [Day, Weekend, PrevSales])
    features = [[weekday, is_weekend, prev_sales]]
    prediction = model.predict(features)
    
    st.header(f"التوقع: {int(prediction[0])} كوب")