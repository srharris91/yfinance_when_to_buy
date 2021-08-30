from stockstats import StockDataFrame

import yfinance as yf
from datetime import date, timedelta
import pandas as pd

import matplotlib.pyplot as plt

today = date.today()
from_days_back = 1000
dstart = (today-timedelta(days=from_days_back)).isoformat()

ticker = 'TQQQ'
history = yf.Ticker(ticker).history(start=dstart,interval='1d',actions=False)

print(history)

stock = StockDataFrame.retype(history)
print(stock)

image_list = ['rsi_6','rsi_12','wr_10','wr_6','macd','close_6_ema','close_6_sma','trix','tema','cci','cci_20','kdjk','dma','pdi','mdi','dx','adx','adxr','cr','tr','atr','boll']
print(stock[image_list])

print(stock)
print(stock.columns)

ax = stock[image_list].plot()
plt.show()
input('Enter to Exit')
