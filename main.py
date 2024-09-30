import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

# Function to fetch stock data and predict prices
def fetch_and_predict(symbol):
    # Fetch historical data
    data = yf.download(symbol, start='2020-01-01', end='2024-01-01')

    # Prepare the data for prediction
    data['Date'] = data.index.map(pd.Timestamp.toordinal)  # Convert dates to ordinal
    X = data['Date'].values.reshape(-1, 1)  # Features (dates)
    y = data['Close'].values  # Target (closing prices)

    # Train a Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Use today's date for prediction
    current_date = pd.Timestamp.today().toordinal()
    future_dates = np.array([current_date + i for i in range(1, 31)]).reshape(-1, 1)
    predicted_prices = model.predict(future_dates)

    # Prepare future dates for plotting
    future_dates_plot = pd.date_range(start=pd.Timestamp.today() + pd.Timedelta(days=1), periods=30)

    # Create a DataFrame for future predictions
    prediction_df = pd.DataFrame({
        'Date': future_dates_plot,
        'Predicted Price': predicted_prices
    })

    # Calculate percentage returns
    prediction_df['Percentage Return'] = prediction_df['Predicted Price'].pct_change() * 100

    # Generate a unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(os.path.expanduser("~"), "Documents", f"{symbol}_predictions_{timestamp}.csv")
    
    # Save to CSV
    prediction_df.to_csv(csv_filename, index=False)

    return data, future_dates_plot, predicted_prices, prediction_df, csv_filename

# Streamlit UI
st.title("Stock Price Predictor")

# Create a dropdown menu for stock selection
stock_var = st.selectbox("Select a stock:", ('AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'))

# Create a button to fetch data and plot
if st.button("Fetch and Predict Stock Price"):
    # Fetch data and predictions
    historical_data, future_dates_plot, predicted_prices, prediction_df, csv_filename = fetch_and_predict(stock_var)

    # Display the historical data
    st.write("### Historical Prices")
    st.line_chart(historical_data['Close'])

    # Plotting predicted prices
    plt.figure(figsize=(12, 6))
    plt.plot(historical_data['Close'], label='Historical Prices', color='blue')
    plt.plot(future_dates_plot, predicted_prices, label='Predicted Prices', color='orange')
    plt.title(f'Stock Price Prediction for {stock_var}')
    plt.xlabel('Date')
    plt.ylabel('Price')

    # Format x-axis ticks to show only the date
    plt.xticks(future_dates_plot, [date.strftime('%Y-%m-%d') for date in future_dates_plot], rotation=45)

    plt.legend()
    plt.grid()
    plt.tight_layout()
    
    # Display the plot in Streamlit
    st.pyplot(plt)

    # Provide a download link for the CSV
    st.write(f"### Predictions saved to {csv_filename}")
    st.download_button(label="Download Predictions CSV", data=prediction_df.to_csv(index=False), file_name=os.path.basename(csv_filename))

