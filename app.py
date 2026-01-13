import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
import time

st.title("Stock Price Predictor App")

from datetime import datetime, timedelta
DEFAULT_HISTORY_YEARS = 10

stock = "GOOG"
stock = st.text_input("Enter the stock here", stock)

@st.cache_data(ttl=3600, show_spinner=False)
def load_stock_history(ticker: str, years: int) -> pd.DataFrame:
    time.sleep(1)
    try:
        data = yf.download(
            ticker,
            period=f"{years}y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        return data
    except Exception:
        return pd.DataFrame()

stock_data = load_stock_history(stock, DEFAULT_HISTORY_YEARS)

if stock_data.empty:
    st.error("Yahoo Finance returned no rows for this ticker/time range. Try another symbol.")
    st.stop()

if isinstance(stock_data.columns, pd.MultiIndex):
    stock_data.columns = stock_data.columns.get_level_values(0)

if "Close" not in stock_data.columns:
    st.error("Downloaded data did not include a 'Close' column; cannot continue.")
    st.stop()

stock_data = stock_data.ffill().dropna()
splitting_len = int(len(stock_data) * 0.9)  
x_test = pd.DataFrame(stock_data['Close'][splitting_len:])

if len(x_test) <= 100:
    st.error("Not enough closing-price samples after the train/test split. Try expanding the history window.")
    st.stop()

model = load_model("Latest_google_model.keras")
st.subheader("All Data")
st.write(stock_data)

st.subheader('Original Close Price')
figsize = (15, 6)
fig = plt.figure(figsize=figsize)

one_year_data = stock_data[-252:]  

plt.plot(one_year_data.index, one_year_data['Close'], 'b')  
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title(f"Original Close Price for the Last Year ({stock})")
st.pyplot(fig)

st.subheader("Test Close Price")
st.write(x_test)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(x_test[['Close']].values)

x_data = []
y_data = []
for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i - 100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

predictions = model.predict(x_data)
inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

ploting_data = pd.DataFrame(
    {
        'original_test_data': inv_y_test.reshape(-1),
        'predictions': inv_pre.reshape(-1)
    },
    index=stock_data.index[splitting_len + 100:]
)
st.subheader("Original values vs Predicted values")
st.write(ploting_data)


st.subheader('Original Close Price vs Predicted Close Price')

one_year_data = stock_data[-252:]  

predicted_values = inv_pre.reshape(-1)
test_data_start = stock_data.index[splitting_len + 100:]


fig = plt.figure(figsize=(15, 6))
plt.plot(one_year_data.index, one_year_data['Close'], label="Original Test Data", color='b')
plt.plot(test_data_start, predicted_values[:len(test_data_start)], label="Predicted Test Data", color='r')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title(f'Original vs Predicted Close Price for the last year ({stock})')
st.pyplot(fig)


st.subheader("Future Price values")

last_100 = stock_data[['Close']].tail(100)
last_100 = scaler.fit_transform(last_100['Close'].values.reshape(-1, 1)).reshape(1, -1, 1)
prev_100 = np.copy(last_100)


def predict_future(no_of_days, prev_100):
    future_predictions = []
    prev_100 = prev_100.reshape(1, 100, 1)  
    for i in range(int(no_of_days)):
        next_day = model.predict(prev_100)  
        future_predictions.append(scaler.inverse_transform(next_day))  
        prev_100 = np.append(prev_100[:, 1:, :], next_day.reshape(1, 1, 1), axis=1)  
    return future_predictions

no_of_days = int(st.text_input("Enter the No of days to be predicted from current date : ", "10"))
future_results = predict_future(no_of_days, prev_100)
future_results = np.array(future_results).reshape(-1, 1)

fig = plt.figure(figsize=(15, 6))
plt.plot(pd.DataFrame(future_results), marker='o')
for i in range(len(future_results)):
    plt.text(i, future_results[i], int(future_results[i][0]))
plt.xlabel('days')
plt.ylabel('Close Price')
plt.title('Closing Price')
st.pyplot(fig)