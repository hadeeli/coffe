import streamlit as st
import pandas as pd
import pickle

# تحميل الموديل
model = pickle.load(open('model.pkl', 'rb'))

st.title("Cups Forecast ☕")

# اختيار التاريخ
selected_date = st.date_input("Select start date")

# تحميل البيانات الأصلية (لازم تكون عندك)
df = pd.read_csv("data.csv", parse_dates=['Date'], index_col='Date')

df_ml = df.copy()

features = ['MA_7', 'MA_30', 'day_of_week', 'month', 'is_weekend',
            'lag_1', 'lag_7', 'rolling_mean_7', 'rolling_std_7']

# زر تشغيل
if st.button("Predict 7 Days"):

    df_future = df_ml.copy()
    future_predictions = []

    # نحول التاريخ المختار
    current_date = pd.to_datetime(selected_date)

    # نتأكد عندنا بيانات قبل التاريخ
    df_future = df_future[df_future.index <= current_date]

    for i in range(7):

        last_row = df_future.iloc[-1:]
        next_date = df_future.index[-1] + pd.Timedelta(days=1)

        new_row = pd.DataFrame(index=[next_date])

        # features زمنية
        new_row['day_of_week'] = next_date.dayofweek
        new_row['month'] = next_date.month
        new_row['is_weekend'] = 1 if next_date.dayofweek >= 5 else 0

        # lag
        new_row['lag_1'] = df_future['Cups_Count'].iloc[-1]
        new_row['lag_7'] = df_future['Cups_Count'].iloc[-7]

        # rolling
        new_row['rolling_mean_7'] = df_future['Cups_Count'].iloc[-7:].mean()
        new_row['rolling_std_7'] = df_future['Cups_Count'].iloc[-7:].std()

        # MA
        new_row['MA_7'] = df_future['Cups_Count'].iloc[-7:].mean()
        new_row['MA_30'] = df_future['Cups_Count'].iloc[-30:].mean()

        # prediction
        pred = model.predict(new_row[features])[0]

        new_row['Cups_Count'] = pred

        future_predictions.append(pred)

        df_future = pd.concat([df_future, new_row])

    # عرض النتائج
    future_dates = pd.date_range(start=current_date + pd.Timedelta(days=1), periods=7)

    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Prediction': future_predictions
    })

    st.write(forecast_df)
