import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import base64
from io import BytesIO


# Dictionary mapping ticker symbols to custom display names
asset_names = {
    '^GSPC': 'S&P500',
    '^DJI': 'Dow Jones',
    '^IXIC': 'NASDAQ',
    '^RUT': 'Russell 2000',
    '^FTSE': 'FTSE 100',
    '^N225': 'Nikkei 225',
    'GC=F': 'Gold',
    'SI=F': 'Silver',
    'CL=F': 'Crude Oil',
    'EURUSD=X': 'EURUSD',
    'GBPUSD=X': 'GBPUSD',
    'AUDUSD=X': 'AUDUSD',
    'NZDUSD=X': 'NZDUSD',
    'USDJPY=X': 'USDJPY',
    'USDCHF=X': 'USDCHF',
    'USDCAD=X': 'USDCAD',
    'EURGBP=X': 'EURGBP',
    'EURJPY=X': 'EURJPY',
    'GBPJPY=X': 'GBPJPY',
    'AUDJPY=X': 'AUDJPY',
    'NZDJPY=X': 'NZDJPY',
    'EURAUD=X': 'EURAUD',
    'EURCHF=X': 'EURCHF',
    'EURCAD=X': 'EURCAD',
    'GBPAUD=X': 'GBPAUD',
    'GBPCAD=X': 'GBPCAD',
    'AUDCAD=X': 'AUDCAD',
    'NZDAUD=X': 'NZDAUD'
}

def fetch_data(ticker, years=10):
    """
    Fetch historical stock data for the given ticker for the past 'years' years.
    """
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(years=years)
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def process_data(data):
    """
    Process the data to create a 2D array of normalized price changes for each year,
    padding shorter years with the last available data point.
    """
    data['Date'] = data.index
    data['Year'] = data['Date'].dt.year
    data['DayOfYear'] = data['Date'].dt.dayofyear

    # Get unique years and initialize the 2D array
    years = data['Year'].unique()
    current_year = pd.Timestamp.today().year

    # Filter out the current year from processing
    years_to_process = years[years != current_year]

    prices_array = []
    max_length = 0

    # Process each year's data
    for year in years_to_process:
        year_data = data[data['Year'] == year]
        if len(year_data) > 0:
            year_prices = year_data['Close'].values
            normalized_prices = year_prices / year_prices[0] - 1
            prices_array.append(normalized_prices)
            max_length = max(max_length, len(normalized_prices))

    # Extend shorter years to match the length of the longest year
    for i in range(len(prices_array)):
        if len(prices_array[i]) < max_length:
            last_valid_price = prices_array[i][-1]
            prices_array[i] = np.pad(prices_array[i], (0, max_length - len(prices_array[i])), 'constant', constant_values=last_valid_price)

    # Process current year separately
    current_year_data = data[data['Year'] == current_year]
    if len(current_year_data) > 0:
        current_year_prices = current_year_data['Close'].values
        normalized_current_year = current_year_prices / current_year_prices[0] - 1
    else:
        normalized_current_year = []

    return np.array(prices_array), years_to_process, normalized_current_year

def get_previous_year_trading_dates(data):
    """
    Get the trading dates from the previous year based on the fetched data.
    """
    end_date = data.index[-1]
    start_date = pd.Timestamp(end_date.year - 1, 1, 1)  # Start of the previous year
    previous_year_data = data.loc[(data.index >= start_date) & (data.index <= end_date)]
    previous_year_trading_dates = previous_year_data.index.date.tolist()
    return previous_year_trading_dates

def plot_seasonal_chart(avg_prices, normalized_current_year, previous_year_trading_dates, ticker):
    """
    Plot the seasonal chart of normalized YTD percentage changes for average and current year.
    Apply moving average smoothing to avg_prices before plotting.
    """
    # Apply moving average smoothing (window size 3)
    avg_prices_smoothed = moving_average(avg_prices, window_size=3)

    plt.figure(figsize=(11, 8))

    # Plot average data
    plt.plot(previous_year_trading_dates[:len(avg_prices_smoothed)], avg_prices_smoothed, linewidth=2, color='#3a64b0')

    # Set title at top left
    plt.text(0.02, 0.95, f'Seasonal Chart for {asset_names.get(ticker, ticker)}', transform=plt.gca().transAxes, fontsize=16, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

    # Calculate start date and current date
    current_year = pd.Timestamp.today().year
    start_year = current_year - 10
    start_date = pd.Timestamp(start_year, 1, 1).strftime('%Y')
    current_date = pd.Timestamp.today().strftime('%Y')

    # Add calculated from (start year) up to (current date) at the top left
    plt.text(0.02, 0.90, f'Calculated from {start_date} up to {current_date}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

    # Add branded label at bottom right
    plt.text(0.98, 0.02, 'BESOMEBODYFX.COM', transform=plt.gca().transAxes, color='black', fontsize=14,
             verticalalignment='bottom', horizontalalignment='right')

    plt.grid(True)
    plt.legend()

    # Format y-axis as percentages multiplied by 100
    def percent(x, pos):
        return '{:.0f}%'.format(x * 100)

    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(percent))

    # Format x-axis as dates
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=15))  # To avoid overcrowding of minor ticks
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
    plt.tight_layout()

    # Convert plot to PNG image
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Encode PNG image to base64 string
    img_base64 = base64.b64encode(img.read()).decode('utf-8')
    img_tag = f'<img src="data:image/png;base64,{img_base64}">'

    return img_tag

def moving_average(data, window_size):
    """
    Compute the moving average of data with a given window size.
    """
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size
