from __future__ import annotations

from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

st.set_page_config(
    page_title="Multi-Ticker Stock Price Predictor",
    page_icon="📈",
    layout="wide",
)
st.title("📈 Multi-Ticker Stock Price Predictor")
st.caption(
    "Train a fresh LSTM model per ticker, evaluate historical performance, and forecast future closes."
)


def fetch_stock_history(ticker: str, history_years: int) -> pd.DataFrame:
    """Download and clean OHLCV data for the requested ticker."""

    end = datetime.now()
    start = end - timedelta(days=history_years * 365)
    data = yf.download(ticker, start=start, end=end, progress=False)
    if data.empty:
        raise ValueError("Yahoo Finance did not return any rows for this ticker/period.")
    cleaned = data[["Open", "High", "Low", "Close", "Volume"]].dropna()
    cleaned.columns = [col.lower() for col in cleaned.columns]
    return cleaned


def make_supervised_sequences(close_prices: pd.Series, lookback: int):
    """Transform a close-price series into LSTM-ready sequences."""

    if len(close_prices) <= lookback:
        raise ValueError("Lookback window is larger than the available data.")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(close_prices.values.reshape(-1, 1))

    sequences, targets = [], []
    for idx in range(lookback, len(scaled_values)):
        sequences.append(scaled_values[idx - lookback : idx])
        targets.append(scaled_values[idx])

    return np.array(sequences), np.array(targets), scaler


def build_lstm_model(lookback: int) -> Sequential:
    """Create a compact LSTM architecture for univariate forecasting."""

    model = Sequential(
        [
            LSTM(128, return_sequences=True, input_shape=(lookback, 1)),
            Dropout(0.2),
            LSTM(64),
            Dense(32, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def train_ticker_model(
    ticker: str,
    history_years: int,
    lookback: int,
    epochs: int,
    batch_size: int,
):
    """End-to-end training loop for a single ticker."""

    data = fetch_stock_history(ticker, history_years)
    close_series = data["close"]
    x, y, scaler = make_supervised_sequences(close_series, lookback)
    if len(x) < 200:
        raise ValueError("Not enough samples for training. Try reducing lookback or increasing history.")

    split_idx = int(len(x) * 0.9)
    x_train, y_train = x[:split_idx], y[:split_idx]
    x_test, y_test = x[split_idx:], y[split_idx:]

    model = build_lstm_model(lookback)
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test) if len(x_test) else None,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )

    return {
        "model": model,
        "history": history.history,
        "scaler": scaler,
        "x_test": x_test,
        "y_test": y_test,
        "close_series": close_series,
        "raw_data": data,
        "lookback": lookback,
    }


def plot_history(history: dict[str, list[float]]):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.get("loss", []), label="Training loss", color="#1f77b4")
    if history.get("val_loss"):
        ax.plot(history["val_loss"], label="Validation loss", color="#ff7f0e")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training Progress")
    ax.legend()
    ax.grid(alpha=0.3)
    return fig


def predict_future_prices(bundle, days_ahead: int) -> pd.DataFrame:
    close_series = bundle["close_series"]
    scaler = bundle["scaler"]
    lookback = bundle["lookback"]
    model = bundle["model"]

    scaled_close = scaler.transform(close_series.values.reshape(-1, 1))
    window = scaled_close[-lookback:].reshape(1, lookback, 1)

    future_prices = []
    for _ in range(days_ahead):
        next_scaled = model.predict(window, verbose=0)
        next_price = scaler.inverse_transform(next_scaled)[0][0]
        future_prices.append(next_price)
        window = np.append(window[:, 1:, :], next_scaled.reshape(1, 1, 1), axis=1)

    future_dates = pd.bdate_range(start=close_series.index[-1] + pd.Timedelta(days=1), periods=days_ahead)
    return pd.DataFrame({"date": future_dates, "predicted_close": future_prices}).set_index("date")

with st.sidebar:
    st.header("Configuration")
    ticker = st.text_input("Ticker symbol", value="GOOG").strip().upper()
    history_years = st.slider("Years of history", min_value=3, max_value=15, value=10)
    lookback = st.slider("Lookback window (days)", min_value=30, max_value=200, value=100, step=10)
    epochs = st.slider("Training epochs", min_value=5, max_value=50, value=10, step=1)
    batch_size = st.select_slider("Batch size", options=[8, 16, 32, 64], value=32)
    future_days = st.slider("Future business days to forecast", min_value=3, max_value=30, value=10)
    retrain = st.button("Train / Refresh model", use_container_width=True)

if not ticker:
    st.info("Enter a ticker symbol to begin.")
    st.stop()

bundle_signature = (ticker, history_years, lookback, epochs, batch_size)
should_train = (
    retrain
    or "model_bundle" not in st.session_state
    or st.session_state.get("bundle_signature") != bundle_signature
)

if should_train:
    try:
        with st.spinner(f"Training LSTM model for {ticker}..."):
            model_bundle = train_ticker_model(
                ticker=ticker,
                history_years=history_years,
                lookback=lookback,
                epochs=epochs,
                batch_size=batch_size,
            )
        st.session_state["model_bundle"] = model_bundle
        st.session_state["bundle_signature"] = bundle_signature
        st.success(f"{ticker} model ready!", icon="✅")
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

model_bundle = st.session_state["model_bundle"]
data = model_bundle["raw_data"]

st.subheader(f"Recent OHLCV data for {ticker}")
st.dataframe(data.tail(10))

col_price, col_volume = st.columns(2)
with col_price:
    fig, ax = plt.subplots(figsize=(8, 4))
    data["close"].plot(ax=ax, color="#1f77b4")
    ax.set_title(f"{ticker} close price ({history_years}y history)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close")
    ax.grid(alpha=0.3)
    st.pyplot(fig)

with col_volume:
    fig, ax = plt.subplots(figsize=(8, 4))
    data["volume"].plot(ax=ax, color="#ff7f0e")
    ax.set_title(f"{ticker} volume")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volume")
    ax.grid(alpha=0.3)
    st.pyplot(fig)
st.subheader("Model diagnostics")
diag_col1, diag_col2 = st.columns(2)

with diag_col1:
    st.pyplot(plot_history(model_bundle["history"]))

with diag_col2:
    x_test = model_bundle["x_test"]
    y_test = model_bundle["y_test"]
    if len(x_test):
        preds_scaled = model_bundle["model"].predict(x_test, verbose=0)
        scaler = model_bundle["scaler"]
        preds = scaler.inverse_transform(preds_scaled)
        y_true = scaler.inverse_transform(y_test)
        idx_start = len(model_bundle["close_series"]) - len(y_true)
        eval_index = model_bundle["close_series"].index[idx_start:]
        comparison = pd.DataFrame(
            {"Actual": y_true.reshape(-1), "Predicted": preds.reshape(-1)},
            index=eval_index,
        )
        st.write("Last 200 evaluation points", comparison.tail(200))

        fig, ax = plt.subplots(figsize=(10, 4))
        comparison.tail(200).plot(ax=ax)
        ax.set_title("Actual vs predicted close price")
        ax.set_xlabel("Date")
        ax.set_ylabel("USD")
        ax.grid(alpha=0.3)
        st.pyplot(fig)
    else:
        st.info("Not enough hold-out samples for evaluation. Increase history length.")
st.subheader(f"Forecasted close price for the next {future_days} business days")
forecast_df = predict_future_prices(model_bundle, future_days)
st.dataframe(forecast_df)

fig, ax = plt.subplots(figsize=(10, 4))
forecast_df["predicted_close"].plot(marker="o", ax=ax)
ax.set_ylabel("USD")
ax.set_xlabel("Date")
ax.set_title(f"{ticker} forecast")
ax.grid(alpha=0.3)
st.pyplot(fig)

st.info(
    "Tip: Adjust lookback, epochs, and history window in the sidebar to retrain a ticker-specific model."
)
