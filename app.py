import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
import os

from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# --------------------------------------------------
# Page config
# --------------------------------------------------

st.set_page_config(page_title="AI Stock Analyzer", layout="wide")

st.title("📈 AI Stock Trend Prediction")

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.header("Stock Search")

ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")

timeframe = st.sidebar.selectbox(
    "Select Timeframe",
    ["1 Month", "6 Months", "1 Year", "5 Years", "Max"]
)

# --------------------------------------------------
# Load Stock Data (with caching)
# --------------------------------------------------

@st.cache_data
def load_data(ticker):
    try:
        data = yf.download(
            ticker,
            start="2010-01-01",
            end="2025-12-31",
            auto_adjust=False
        )
        return data
    except:
        return None


df = load_data(ticker)

if df is None or df.empty:
    st.error("⚠️ Stock data unavailable (Yahoo rate limit). Refresh the app.")
    st.stop()

# --------------------------------------------------
# Timeframe filtering
# --------------------------------------------------

if timeframe == "1 Month":
    df_chart = df.tail(22)

elif timeframe == "6 Months":
    df_chart = df.tail(126)

elif timeframe == "1 Year":
    df_chart = df.tail(252)

elif timeframe == "5 Years":
    df_chart = df.tail(1260)

else:
    df_chart = df

# --------------------------------------------------
# Candlestick chart
# --------------------------------------------------

st.subheader("Candlestick Chart")

fig = go.Figure(data=[go.Candlestick(
    x=df_chart.index,
    open=df_chart["Open"],
    high=df_chart["High"],
    low=df_chart["Low"],
    close=df_chart["Close"]
)])

fig.update_layout(height=500)

st.plotly_chart(fig)

# --------------------------------------------------
# Data summary
# --------------------------------------------------

st.subheader("Stock Data Summary")

st.write(df.describe())

# --------------------------------------------------
# Closing price chart
# --------------------------------------------------

st.subheader("Closing Price vs Time")

fig1 = plt.figure(figsize=(12,6))

plt.plot(df["Close"])

st.pyplot(fig1)

# --------------------------------------------------
# Moving averages
# --------------------------------------------------

st.subheader("Moving Averages (100 & 200)")

ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()

fig2 = plt.figure(figsize=(12,6))

plt.plot(df.Close, label="Close Price")
plt.plot(ma100, label="100 MA")
plt.plot(ma200, label="200 MA")

plt.legend()

st.pyplot(fig2)

# --------------------------------------------------
# Load AI model safely
# --------------------------------------------------

model_path = "keras_model.h5"

if not os.path.exists(model_path):
    st.error("❌ Model file not found!")
    st.stop()

try:
    model = load_model(model_path, compile=False)
    st.success("✅ Model loaded successfully")
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# --------------------------------------------------
# Data preparation
# --------------------------------------------------

data_training = pd.DataFrame(df["Close"][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df["Close"][int(len(df)*0.70):])

scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)

x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)

# --------------------------------------------------
# Testing data
# --------------------------------------------------

past_100_days = data_training.tail(100)

final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)

# --------------------------------------------------
# Prediction
# --------------------------------------------------

y_predicted = model.predict(x_test)

scale = scaler.scale_

y_predicted = y_predicted / scale[0]
y_test = y_test / scale[0]

# --------------------------------------------------
# Prediction chart
# --------------------------------------------------

st.subheader("Prediction vs Original Price")

fig3 = plt.figure(figsize=(12,6))

plt.plot(y_test, label="Original Price")
plt.plot(y_predicted, label="Predicted Price")

plt.legend()

st.pyplot(fig3)

# --------------------------------------------------
# Next day prediction
# --------------------------------------------------

st.subheader("Next Day Predicted Price")

next_price = y_predicted[-1][0]

st.success(f"Predicted Next Price: ${next_price:.2f}")

# --------------------------------------------------
# Prediction accuracy
# --------------------------------------------------

st.subheader("Model Prediction Accuracy")

error = np.mean(np.abs(y_test - y_predicted))

accuracy = max(0, 100 - error)

st.write(f"Average Prediction Error: {error:.2f}")
st.write(f"Prediction Accuracy Score: {accuracy:.2f}%")

# --------------------------------------------------
# Risk indicator
# --------------------------------------------------

st.subheader("Stock Risk Indicator")

df["Daily Return"] = df["Close"].pct_change()

volatility = df["Daily Return"].std() * np.sqrt(252)

st.write(f"Annual Volatility: {volatility:.2f}")

if volatility < 0.2:
    st.success("Low Risk Stock")

elif volatility < 0.4:
    st.warning("Medium Risk Stock")

else:
    st.error("High Risk Stock")

# --------------------------------------------------
# Latest market data
# --------------------------------------------------

st.subheader("Latest Market Data")

st.write(df.tail())
