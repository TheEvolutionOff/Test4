import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import base64
from io import BytesIO
import streamlit as st

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
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(years=years)
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def process_data(data):
    data['Date'] = data.index
    data['Year'] = data['Date'].dt.year
    data['DayOfYear'] = data['Date'].dt.dayofyear

    years = data['Year'].unique()
    current_year = pd.Timestamp.today().year
    years_to_process = years[years != current_year]

    prices_array = []
    max_length = 0

    for year in years_to_process:
        year_data = data[data['Year'] == year]
        if len(year_data) > 0:
            year_prices = year_data['Close'].values
            normalized_prices = year_prices / year_prices[0] - 1
            prices_array.append(normalized_prices)
            max_length = max(max_length, len(normalized_prices))

    for i in range(len(prices_array)):
        if len(prices_array[i]) < max_length:
            last_valid_price = prices_array[i][-1]
            prices_array[i] = np.pad(prices_array[i], (0, max_length - len(prices_array[i])), 'constant', constant_values=last_valid_price)

    current_year_data = data[data['Year'] == current_year]
    if len(current_year_data) > 0:
        current_year_prices = current_year_data['Close'].values
        normalized_current_year = current_year_prices / current_year_prices[0] - 1
    else:
        normalized_current_year = []

    return np.array(prices_array), years_to_process, normalized_current_year

def get_previous_year_trading_dates(data):
    end_date = data.index[-1]
    start_date = pd.Timestamp(end_date.year - 1, 1, 1)
    previous_year_data = data.loc[(data.index >= start_date) & (data.index <= end_date)]
    previous_year_trading_dates = previous_year_data.index.date.tolist()
    return previous_year_trading_dates

def plot_seasonal_chart(avg_prices, normalized_current_year, previous_year_trading_dates, ticker):
    avg_prices_smoothed = moving_average(avg_prices, window_size=3)

    plt.figure(figsize=(11, 8))
    plt.plot(previous_year_trading_dates[:len(avg_prices_smoothed)], avg_prices_smoothed, linewidth=2, color='#3a64b0')

    plt.text(0.02, 0.95, f'Seasonal Chart for {asset_names.get(ticker, ticker)}', transform=plt.gca().transAxes, fontsize=16, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

    current_year = pd.Timestamp.today().year
    start_year = current_year - 10
    start_date = pd.Timestamp(start_year, 1, 1).strftime('%Y')
    current_date = pd.Timestamp.today().strftime('%Y')

    plt.text(0.02, 0.90, f'Calculated from {start_date} up to {current_date}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    plt.text(0.98, 0.02, 'BESOMEBODYFX.COM', transform=plt.gca().transAxes, color='black', fontsize=14, verticalalignment='bottom', horizontalalignment='right')

    plt.grid(True)
    plt.legend()

    def percent(x, pos):
        return '{:.0f}%'.format(x * 100)

    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(percent))
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=15))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    plt.xticks(rotation=45)
    plt.tight_layout()

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    img_base64 = base64.b64encode(img.read()).decode('utf-8')
    img_tag = f'<img src="data:image/png;base64,{img_base64}">'

    return img_tag

def moving_average(data, window_size):
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size

st.title("Stock Analysis Tool")
ticker = st.selectbox("Choose Asset:", list(asset_names.keys()), format_func=lambda x: asset_names[x])

if st.button("Generate Chart"):
    data = fetch_data(ticker)
    prices_array, years, normalized_current_year = process_data(data)
    previous_year_trading_dates = get_previous_year_trading_dates(data)
    avg_prices = np.nanmean(prices_array, axis=0)
    chart = plot_seasonal_chart(avg_prices, normalized_current_year, previous_year_trading_dates, ticker)
    st.markdown(chart, unsafe_allow_html=True)
