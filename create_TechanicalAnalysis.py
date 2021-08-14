import yfinance as yf
import numpy as np

from datetime import date,timedelta
import pandas as pd

import matplotlib.pyplot as plt
plt.style.use('seaborn-notebook')

today = date.today()
dend = today.isoformat()
#print(dstart,dend)
days_back = 100 # earliest day to test 
dstart = (today-timedelta(days=days_back)).isoformat() # 60 days prior to today
 
ticker = 'TQQQ'
history = yf.Ticker(ticker).history(start=dstart,interval='1d') # should give 280 rows of data (40 work days, 7 hours a day)
print(history)
#print(history.index[-4])
#print(history.index[-3])
#print(history.index[-2])
#print(history.index[-1])

rows,cols = history.shape

figtick,axtick = plt.subplots(figsize=(4.75,3),dpi=200,tight_layout=True)
axtick.plot(history.index,history['Close'],label=ticker)
print(ticker + ' price is currently       $',history['Close'][-1],' this hour ',history.index[-1])
axtick.set_ylabel(ticker+' price $')
for tick in axtick.get_xticklabels():
    tick.set_rotation(45)

figtick.show()

# calculate technical analysis (TA) items are oscillator and trend based financial time series filters that are commonly preferred by short to medium term traders (6-20 days ranges, swing trades for 1 week to 1 month periods are focused)
# list of TA desired to calculate as used by recent ML model
# https://www.researchgate.net/publication/324802031_Algorithmic_Financial_Trading_with_Deep_Convolutional_Neural_Networks_Time_Series_to_Image_Conversion_Approach
# list: RSI, Williams %R, WMA, EMA, SMA, HMA, Triple EMA, CCI, CMO, MACD, PPO, ROC, CMFI, DMI, and PSI with intervals of 6 to days
def RSI(data,period=14):
    rows = data.shape[0]
    data['RIS'] = data['Close']
    data.loc[data.index[:period],'RIS'] = np.nan
    for i in range(period,rows):
        # get average gain and average loss for previous period
        d = data.loc[data.index[i-period:i+1],'Close']
        if i==period:
            diff = (d.diff().to_numpy()/d.to_numpy())
            avg_positive = diff[diff>0].mean()
            avg_negative = diff[diff<=0].mean()
            RSI0 = 100.0-(100.0/(1.0+avg_positive/avg_negative))
        else:
            pass
        #avg_gain = 
        #data.loc[data.index[i],'RIS'] = 

RSI(history)
#print(history)



input('Enter to exit')
