import yfinance as yf
import numpy as np

from datetime import date,timedelta
import pandas as pd

import matplotlib.pyplot as plt
plt.style.use('seaborn-notebook')

today = date.today()
dend = today.isoformat()
#print(dstart,dend)
from_days_back = 30 # earliest day to test 
dstart = (today-timedelta(days=from_days_back)).isoformat() # 60 days prior to today
 
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

# now model it like each day, starting amount is $10,000, assume no fees on transactions
def model_growth_of_best(data):
    num_days = data.shape[0]
    net_worth = pd.DataFrame({'net worth':np.zeros(num_days), 'strategy':np.zeros(num_days)},index=data.index)
    net_worth['strategy'] = net_worth['strategy'].astype(str)
    funds = 1000.00
    #price = data['Close'][-from_days_back-1] # day to purchase
    #funds -= price # purchase one share at the beginning
    shares = 0
    for i,day in enumerate(data.index[:-1]):
        if i==0:
            what_to_do = 'buy'
            price = data['Close'][i]
        else:
            d = data[i-1:i+2]
            what_to_do = 'hold'
            argmax = np.argmax(d['Close'])
            argmin = np.argmin(d['Close'])
            if argmax==1:
                what_to_do = 'sell'
            elif argmin==1:
                what_to_do = 'buy'
            else:
                what_to_do = 'hold'
            price = d['Close'][1]
        if what_to_do == 'buy':
            num_shares_to_buy = funds//price
            funds -= num_shares_to_buy*price
            shares += num_shares_to_buy
        elif what_to_do == 'sell':
            funds += shares*price
            shares = 0
        elif what_to_do == 'hold':
            pass
        net_worth.loc[day,'net worth'] = funds + price*shares
        net_worth.loc[day,'strategy'] = what_to_do
    for i,day in enumerate(data.index[-1:]):
        price = data.loc[day,'Close']
        net_worth.loc[day,'net worth'] = funds + price*shares
        net_worth.loc[day,'strategy'] = 'hold'
    # sell at the end
    print('selling shares at the end')
    price = data['Close'][-1]
    funds += price*shares
    shares = 0
    print(funds,shares,price)
    return net_worth

# buy and hold strategy
net_worth_best = model_growth_of_best(history)

# plot strategies net worth
fig,ax = plt.subplots(figsize=(4.75,3),dpi=200,tight_layout=True)
ax.plot(net_worth_best['net worth'],label='best')
def plot_buy_and_sell(net_worth,ax,column='net worth'):
    for i,day in enumerate(net_worth.index):
        if net_worth['strategy'][i] == 'buy':
            ax.plot(net_worth.index[i],net_worth[column][i],'g^')
        elif net_worth['strategy'][i] == 'sell':
            ax.plot(net_worth.index[i],net_worth[column][i],'rv')
net_worth_best['Close'] = history['Close']
plot_buy_and_sell(net_worth_best,axtick,column='Close')
ax.set_xlabel('days')
ax.set_ylabel('net worth $')
for tick in ax.get_xticklabels():
    tick.set_rotation(45)
ax.legend(loc='best',numpoints=1)
axtick.legend(loc='best',numpoints=1)
figtick.show()
fig.show()







days_avg = 14 # how many hours to look back to predict the future

input('Enter to exit')
