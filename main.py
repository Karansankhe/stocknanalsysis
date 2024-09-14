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
        data = stock.history(period="1d")
        # Convert the DataFrame to a dictionary and ensure all keys are strings
        data_dict = data.to_dict()
        data_dict_str_keys = {str(k): {str(inner_k): inner_v for inner_k, inner_v in v.items()} for k, v in data_dict.items()}
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
st.set_page_config(page_title="Stock Data Fetcher", layout="wide")
st.title("Stock Data Fetcher")

# Streamlit inputs
symbol = st.text_input("Enter Stock Symbol")

if st.button("Fetch Data"):
    if symbol:
        data = fetch_yfinance_stock_data(symbol)
        if data is not None and not data.empty:
            st.subheader(f"Stock Data for {symbol}")
            
            # Display raw data
            st.write(data)

            # Visualization
            st.subheader("Stock Price Chart")
            fig, ax = plt.subplots()
            data['Close'].plot(ax=ax, title=f"Closing Prices for {symbol}", legend=True)
            st.pyplot(fig)

            # Fetch and display Gemini response
            gemini_response = get_gemini_response(symbol)
            st.subheader("Gemini Response")
            st.write(gemini_response)
        else:
            st.error(f"Failed to fetch data for {symbol}")
    else:
        st.error("Please enter a stock symbol.")
