import yfinance as yf
import streamlit as st
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import pandas as pd
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Configure the Google Gemini model with API key
api_key = os.getenv('GENAI_API_KEY')
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-pro")

# Function to fetch stock data from Yahoo Finance using yfinance
def fetch_yfinance_stock_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period="1y")  # Fetching 1 year of historical data
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol} from Yahoo Finance: {str(e)}")
        return None

# Function to interact with Gemini AI for stock-related queries
def get_gemini_response(symbol):
    prompt_template = (
        "You are an intelligent assistant with expertise in stock market analysis. "
        "Provide detailed insights or analysis on the following stock:\n\n"
        "Stock: {}\n"
        "Analysis:"
    )

    response = model.generate_content(prompt_template.format(symbol), stream=True)
    full_text = ""
    for chunk in response:
        full_text += chunk.text
    return full_text

# Streamlit app setup
st.set_page_config(page_title="Stock Data Dashboard", layout="wide")
st.title("Stock Data Dashboard")

# Streamlit inputs
symbols = st.multiselect("Select Stock Symbols", ["RELIANCE.BO", "TCS.BO", "HDFCBANK.BO", "INFY.BO", "ICICIBANK.BO", 
    "HINDUNILVR.BO", "BHARTIARTL.BO", "ITC.BO", "KOTAKBANK.BO", "SBI.BO",
    "LTI.BO", "WIPRO.BO", "HCLTECH.BO", "M&M.BO", "ADANIGREEN.BO", "NTPC.BO",
    "POWERGRID.BO", "ONGC.BO", "BAJFINANCE.BO", "JSWSTEEL.BO", "HDFC.BO",
    "M&MFIN.BO", "SBILIFE.BO", "CIPLA.BO", "DRREDDY.BO", "SUNPHARMA.BO", "TSLA"], default=["RELIANCE.BO"])

if st.button("Fetch Data"):
    if symbols:
        for symbol in symbols:
            data = fetch_yfinance_stock_data(symbol)
            if data is not None and not data.empty:
                st.subheader(f"Stock Data for {symbol}")
                
                # Display raw data
                st.write("**Displaying last 5 records:**")
                st.write(data.tail())
                
                # Create a column layout for visualizations
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Stock Price Chart
                    st.subheader(f"Stock Price Chart for {symbol}")
                    fig, ax = plt.subplots(figsize=(4, 3))  # Smaller figure size
                    data['Close'].plot(ax=ax, title=f"Closing Prices for {symbol}", legend=True)
                    plt.xlabel("Date")
                    plt.ylabel("Price (USD)")
                    st.pyplot(fig)
                
                with col2:
                    # Volume Traded Chart
                    st.subheader(f"Volume Traded Chart for {symbol}")
                    fig, ax = plt.subplots(figsize=(4, 3))  # Smaller figure size
                    data['Volume'].plot(ax=ax, color='orange', title=f"Volume Traded for {symbol}", legend=True)
                    plt.xlabel("Date")
                    plt.ylabel("Volume")
                    st.pyplot(fig)
                
                with col3:
                    # Moving Average Chart
                    st.subheader(f"Moving Average Chart for {symbol}")
                    fig, ax = plt.subplots(figsize=(4, 3))  # Smaller figure size
                    data['Close'].rolling(window=30).mean().plot(ax=ax, color='blue', label='30-Day Moving Average')
                    data['Close'].plot(ax=ax, title=f"Closing Prices and Moving Average for {symbol}", legend=True)
                    plt.xlabel("Date")
                    plt.ylabel("Price (USD)")
                    st.pyplot(fig)

                # Fetch and display Gemini response
                gemini_response = get_gemini_response(symbol)
                st.subheader(f"Gemini Response for {symbol}")
                st.write(gemini_response)
            else:
                st.error(f"Failed to fetch data for {symbol}")
    else:
        st.error("Please select at least one stock symbol.")
