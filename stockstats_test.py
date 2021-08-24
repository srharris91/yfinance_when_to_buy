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

print(stock['macd'])

print(stock)

ax = stock.plot()
plt.show()
input('Enter to Exit')
