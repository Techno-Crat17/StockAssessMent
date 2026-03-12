import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
import time

st.set_page_config(page_title="Stock Analyzer", layout="wide")

st.title("📈 Stock Market Analyzer")

# -----------------------------------
# Sidebar
# -----------------------------------

st.sidebar.header("Stock Search")

ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")

timeframe = st.sidebar.selectbox(
    "Select Timeframe",
    ["1 Month","6 Months","1 Year","5 Years","Max"]
)

# -----------------------------------
# Load Data with Retry
# -----------------------------------

@st.cache_data(ttl=3600)
def load_data(ticker):

    for i in range(3):  # retry 3 times
        try:
            df = yf.download(
                ticker,
                start="2010-01-01",
                end="2025-12-31",
                auto_adjust=False
            )

            if not df.empty:
                df.columns = df.columns.get_level_values(0)
                return df

        except:
            time.sleep(2)

    return None


df = load_data(ticker)

if df is None or df.empty:
    st.error("⚠️ Yahoo Finance rate limit reached. Please refresh in a few seconds.")
    st.stop()

# -----------------------------------
# Timeframe Filter
# -----------------------------------

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

# -----------------------------------
# Candlestick Chart
# -----------------------------------

st.subheader("Candlestick Chart")

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df_chart.index,
    open=df_chart["Open"],
    high=df_chart["High"],
    low=df_chart["Low"],
    close=df_chart["Close"]
))

fig.update_layout(
    height=500,
    xaxis_title="Date",
    yaxis_title="Price"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------------
# Moving Averages
# -----------------------------------

st.subheader("Moving Average (100 & 200)")

ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()

fig2 = plt.figure(figsize=(12,6))

plt.plot(df.Close,label="Close")
plt.plot(ma100,label="100 MA")
plt.plot(ma200,label="200 MA")

plt.legend()

st.pyplot(fig2)

# -----------------------------------
# Risk Indicator
# -----------------------------------

st.subheader("Stock Risk Indicator")

df["Daily Return"] = df["Close"].pct_change()

volatility = df["Daily Return"].std()*np.sqrt(252)

st.write(f"Annual Volatility: {volatility:.2f}")

if volatility < 0.2:
    st.success("Low Risk")

elif volatility < 0.4:
    st.warning("Medium Risk")

else:
    st.error("High Risk")

# -----------------------------------
# Latest Market Data
# -----------------------------------

st.subheader("Latest Market Data")

st.write(df.tail())
