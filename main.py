import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from datetime import datetime
import os

def fetch_and_predict(symbol):
    data = yf.download(symbol, start='2020-01-01', end='2024-01-01')
    data['Date'] = data.index.map(pd.Timestamp.toordinal)
    X = data['Date'].values.reshape(-1, 1)
    y = data['Close'].values
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    current_date = pd.Timestamp.today().toordinal()
    future_dates = np.array([current_date + i for i in range(1, 31)]).reshape(-1, 1)
    predicted_prices = model.predict(future_dates)
    future_dates_plot = pd.date_range(start=pd.Timestamp.today() + pd.Timedelta(days=1), periods=30)
    prediction_df = pd.DataFrame({
        'Date': future_dates_plot,
        'Predicted Price': predicted_prices
    })
    prediction_df['Percentage Return'] = prediction_df['Predicted Price'].pct_change() * 100
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(os.path.expanduser("~"), "Documents", f"{symbol}_predictions_{timestamp}.csv")
    prediction_df.to_csv(csv_filename, index=False)
    return data, future_dates_plot, predicted_prices, prediction_df, csv_filename

st.title("Stock Price Predictor")
stock_var = st.selectbox("Select a stock:", ('AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'))

if st.button("Fetch and Predict Stock Price"):
    historical_data, future_dates_plot, predicted_prices, prediction_df, csv_filename = fetch_and_predict(stock_var)
    st.write("### Historical Prices")
    st.line_chart(historical_data['Close'])
    plt.figure(figsize=(12, 6))
    plt.plot(historical_data['Close'], label='Historical Prices', color='blue')
    plt.plot(future_dates_plot, predicted_prices, label='Predicted Prices', color='orange')
    plt.title(f'Stock Price Prediction for {stock_var}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.xticks(future_dates_plot, [date.strftime('%Y-%m-%d') for date in future_dates_plot], rotation=45)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    st.pyplot(plt)
    st.write(f"### Predictions saved to {csv_filename}")
    st.download_button(label="Download Predictions CSV", data=prediction_df.to_csv(index=False), file_name=os.path.basename(csv_filename))
