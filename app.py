import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import time

from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("📈 Stock Price Predictor App")

# ---------------- USER INPUT ----------------
stock = st.text_input(
    "Enter stock symbol (GOOGL recommended for Google):",
    "GOOG"
).upper()

# ---------------- SAFE DATA LOADER WITH FALLBACK ----------------
@st.cache_data(ttl=3600)
def load_stock_data(symbol):
    """
    Yahoo Finance loader with automatic GOOG -> GOOGL fallback
    Works reliably on Streamlit Cloud
    """
    time.sleep(1)  # prevent rate limiting

    symbols_to_try = [symbol]

    # Auto fallback for Google
    if symbol == "GOOG":
        symbols_to_try.append("GOOGL")

    for sym in symbols_to_try:
        try:
            data = yf.download(
                sym,
                period="10y",
                interval="1d",
                progress=False,
                threads=False
            )
            if not data.empty:
                return data, sym
        except Exception:
            continue

    return pd.DataFrame(), None

# ---------------- LOAD DATA ----------------
stock_data, used_symbol = load_stock_data(stock)

if stock_data.empty:
    st.error(
        "❌ Yahoo Finance blocked this ticker on Streamlit Cloud.\n\n"
        "✔ Try AAPL, MSFT, TSLA\n"
        "✔ Or use GOOGL instead of GOOG"
    )
    st.stop()

if used_symbol != stock:
    st.info(f"ℹ️ Data loaded using alternate ticker: **{used_symbol}**")

# Ensure Close column exists
if "Close" not in stock_data.columns:
    st.error("Downloaded data does not contain 'Close' prices.")
    st.stop()

stock_data = stock_data.ffill().dropna()

# ---------------- LOAD MODEL ----------------
model = load_model("Latest_google_model.keras")

# ---------------- DISPLAY DATA ----------------
st.subheader("📊 Stock Data (Last 10 Years)")
st.dataframe(stock_data.tail(300))

# ---------------- TRAIN / TEST SPLIT ----------------
splitting_len = int(len(stock_data) * 0.9)
test_data = stock_data[['Close']].iloc[splitting_len:]

if len(test_data) <= 100:
    st.error("Not enough data for prediction.")
    st.stop()

# ---------------- SCALING ----------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_test = scaler.fit_transform(test_data)

x_test, y_test = [], []
for i in range(100, len(scaled_test)):
    x_test.append(scaled_test[i - 100:i])
    y_test.append(scaled_test[i])

x_test = np.array(x_test)
y_test = np.array(y_test)

# ---------------- PREDICTION ----------------
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
actual = scaler.inverse_transform(y_test)

# ---------------- PLOT TEST RESULTS ----------------
st.subheader("📉 Actual vs Predicted Close Price")

fig = plt.figure(figsize=(15, 6))
plt.plot(actual, label="Actual Price", color="blue")
plt.plot(predictions, label="Predicted Price", color="red")
plt.legend()
plt.xlabel("Days")
plt.ylabel("Close Price")
plt.title(f"{used_symbol} Price Prediction")
st.pyplot(fig)

# ---------------- FUTURE PREDICTION ----------------
st.subheader("🔮 Future Price Prediction")

days = st.number_input(
    "Enter number of days to predict:",
    min_value=1,
    max_value=30,
    value=10
)

last_100 = stock_data[['Close']].tail(100)
last_100_scaled = scaler.fit_transform(last_100).reshape(1, 100, 1)

future_predictions = []
current_input = last_100_scaled

for _ in range(days):
    next_price = model.predict(current_input)
    future_predictions.append(
        scaler.inverse_transform(next_price)[0][0]
    )
    current_input = np.append(
        current_input[:, 1:, :],
        next_price.reshape(1, 1, 1),
        axis=1
    )

# ---------------- FUTURE PLOT WITH VALUES ----------------
fig = plt.figure(figsize=(12, 5))
plt.plot(future_predictions, marker="o")

for i, value in enumerate(future_predictions):
    plt.text(
        i,
        value,
        f"{value:.2f}",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.xlabel("Days")
plt.ylabel("Predicted Close Price")
plt.title(f"{used_symbol} – Next {days} Days Prediction")
st.pyplot(fig)

st.success("✅ Prediction completed successfully!")
