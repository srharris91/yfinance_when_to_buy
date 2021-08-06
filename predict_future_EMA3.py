import torch
import yfinance as yf
import numpy as np
import copy

from datetime import date,timedelta
import pandas as pd

import matplotlib.pyplot as plt
plt.style.use('seaborn-notebook')

today = date.today()
dend = today.isoformat()
dstart = (today-timedelta(days=20)).isoformat() # 60 days prior to today
#print(dstart,dend)

ticker = 'TQQQ'
history = yf.Ticker(ticker).history(start=dstart,interval='1d') # should give 280 rows of data (40 work days, 7 hours a day)
#print(history.index[-4])
#print(history.index[-3])
#print(history.index[-2])
#print(history.index[-1])

rows,cols = history.shape
hours_back = 14 # how many hours to look back to predict the future
N = 200 # how many samples
cols = 2 # always 7 columns ['Open','High','Low','Close','Volume','Dividends','Stock Splits'] # unless you only want Close and Volume
# we want to predict the next hour 'Close' and 'Volume'
D_in = cols
H = 100 # hidden layers
D_out = 2 # predict 'Close' and 'Volume price at next hour

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
print('device = ',device)
def EMA(x):
    '''Exponential moving average
    x_{t+1} = EMA_t = γ*EMA_{t-1} + (1-γ)*x_t
    Inputs:
        x: inputs data of shape (time series,number of variables)
    Outputs:
        out: variables at the next time of shape (time series+1,number of variables)
    '''
    N,nvars = x.shape
    EMA = np.zeros((N+1,nvars))
    γ = 0.1 # how much of the current EMA value to consider
    for i,xi in enumerate(x):
        EMA[i+1] = γ*EMA[i] + (1.0-γ)*xi
    return EMA

model = lambda x: EMA(x)
        
print(model)

#with torch.no_grad():
xin = history[['Close','Volume']].values
y = history[['Close','Volume']].values
y_pred = model(xin)

fig,ax = plt.subplots(figsize=(4.75,3),dpi=200,tight_layout=True)
ax.plot(history.index,history['Close'],label=ticker)
ax.plot(history.index[1:],y_pred[1:-1,0],'.',label=ticker+' model')
print(ticker + ' price is currently       $',history['Close'][-1],' this hour ',history.index[-1])
ax.set_ylabel(ticker+' price $')
for tick in ax.get_xticklabels():
    tick.set_rotation(45)
# predict what it'll be next hour
print(ticker + ' price is predicted to be $',y_pred[-1,0],' next hour ',history.index[-1]+pd.to_timedelta(1,unit='h'))
ax.plot(history.index[-1]+pd.to_timedelta(1,unit='h'),y_pred[-1,0],'.',label=ticker+' pred for next hour')
ax.legend(loc='best',numpoints=1)
fig.show()

input('Enter to exit')
