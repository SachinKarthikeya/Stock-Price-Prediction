import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import time
import requests
import matplotlib.pyplot as plt

# Load the saved model and scaler
model = load_model('/Users/sachinkarthikeya/Desktop/Projects/SPP-LSTM/stock_prediction_model.h5')
scaler = joblib.load('/Users/sachinkarthikeya/Desktop/Projects/SPP-LSTM/scaler.pkl')

# Load and preprocess data
stock_data = pd.read_csv('/Users/sachinkarthikeya/Desktop/Projects/SPP-LSTM/preprocessed_stock_data.csv')

def predict_next_7_days(company_symbol):
    if 'Symbol_encoded' not in stock_data.columns:
        company_ids = stock_data['Symbol'].astype('category').cat.codes
        stock_data['Symbol_encoded'] = company_ids

    company_data = stock_data[stock_data['Symbol'] == company_symbol]
    if company_data.empty:
        raise ValueError(f"No data found for the company symbol: {company_symbol}")
    
    company_id = company_data['Symbol_encoded'].iloc[0]
    last_90_days = company_data[['close', 'volume', '20_MA', '50_MA']].tail(90).values
    if len(last_90_days) < 90:
        raise ValueError("Not enough data to make predictions.")

    last_90_days_scaled = scaler.transform(last_90_days)
    prediction_input = np.hstack([last_90_days_scaled, np.full((90, 1), company_id)])
    prediction_input = np.expand_dims(prediction_input, axis=0)

    predictions = []
    for _ in range(7):  # Predict next 7 days
        pred = model.predict(prediction_input)[0][0]  # Predict one step ahead
        predictions.append(pred)

        # Update input with the new prediction
        new_input = np.append(prediction_input[0][1:], [[pred, 0, 0, 0, company_id]], axis=0)
        prediction_input = np.expand_dims(new_input, axis=0)

    # Inverse transform the predictions
    predictions = scaler.inverse_transform(
        np.hstack([np.array(predictions).reshape(-1, 1), np.zeros((7, 3))])
    )[:, 0]

    return predictions

def fetch_news():
    api_key = "1056cf9def0944eea9e58647b7548b1d"  # Replace with your NewsAPI key
    url = f"https://newsapi.org/v2/top-headlines?category=business&apiKey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json()["articles"]
        return [{"title": article["title"], "url": article["url"]} for article in articles[:5]]
    else:
        st.error("Failed to fetch news.")
        return []

# Streamlit UI
st.title("Stock Price Prediction Dashboard")
st.write("Predict the next 7 days' stock prices for your desired company.")

# User Input
company_symbol = st.text_input("Enter the company symbol:", value="")

if st.button("Predict"):
    try:
        if company_symbol.strip():
            predicted_prices = predict_next_7_days(company_symbol.strip())
            st.success(f"Predicted prices for the next 7 days for '{company_symbol}':")
            st.write(predicted_prices)

            # Plotting the predictions
            st.write("### Predicted Prices Visualization")
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, 8), predicted_prices, marker='o', linestyle='-', color='b', label="Predicted Prices")
            plt.xlabel("Days")
            plt.ylabel("Price")
            plt.title(f"Predicted Stock Prices for '{company_symbol}'")
            plt.legend()
            plt.grid()
            st.pyplot(plt)
        else:
            st.error("Please enter a valid company symbol.")
    except Exception as e:
        st.error(f"Error: {e}")

# Display Latest News
st.sidebar.title("Latest Stock Market News")
news = fetch_news()
if news:
    for article in news:
        st.sidebar.write(f"### [{article['title']}]({article['url']})")

refresh_interval = 60 
st.sidebar.write(f"**Note:** News refreshes every {refresh_interval} seconds.")
time.sleep(refresh_interval)
st.experimental_rerun()