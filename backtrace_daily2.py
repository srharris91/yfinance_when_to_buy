import yfinance as yf
import numpy as np

from datetime import date,timedelta
import pandas as pd

import matplotlib.pyplot as plt
plt.style.use('seaborn-notebook')

today = date.today()
dend = today.isoformat()
#print(dstart,dend)
from_mins_back = 1000 # earliest day to test 
to_mins_back = 1 # latest day to test, 1 means yesterday
dstart = (today-timedelta(days=6)).isoformat() # 60 days prior to today
 
ticker = 'TQQQ'
history = yf.Ticker(ticker).history(start=dstart,interval='1m') # should give 280 rows of data (40 work days, 7 hours a day)
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
# predict what it'll be next hour

# now attempt to model growth from a model that says to buy or sell
def to_buy_hold_sell_MA(data,period='15'):
    ''' model to tell you to buy hold or sell, will return a string matching that
    Inputs:
        data: dataframe with Close values from previous days
    Outputs:
        str: buy, sell, or hold
    '''
    # buy or sell if the rolling average of yesterday was the local min or max
    rolling_close = data['rollingClose'+period][-3:] # take last 3 days
    argmax = np.argmax(rolling_close)
    argmin = np.argmin(rolling_close)
    what_to_do = 'hold'
    if argmax == 1:
        what_to_do = 'sell'
    elif argmin == 1:
        what_to_do = 'buy'
    else:
        what_to_do = 'hold'
    return what_to_do
    print(argmax,argmin)
def to_buy_and_hold(data):
    ''' model to tell you to buy and hold
    Inputs:
        data: dataframe with Close values from previous days
    Outputs:
        str: hold
    '''
    # buy or sell if the rolling average of yesterday was the local min or max
    what_to_do = 'hold'
    return what_to_do


# preprocess data to have rollingClose data of rolling average of past 25 days
history['rollingClose3'] = history['Close'].rolling(3).mean()
history['rollingClose5'] = history['Close'].rolling(5).mean()
history['rollingClose10'] = history['Close'].rolling(10).mean()
history['rollingClose15'] = history['Close'].rolling(15).mean()
history['rollingClose25'] = history['Close'].rolling(25).mean()
history['rollingClose50'] = history['Close'].rolling(50).mean()
axtick.plot(history.index,history['rollingClose3'],label=ticker + ' 3 MA')
axtick.plot(history.index,history['rollingClose5'],label=ticker + ' 5 MA')
axtick.plot(history.index,history['rollingClose10'],label=ticker + ' 10 MA')
axtick.plot(history.index,history['rollingClose15'],label=ticker + ' 15 MA')
axtick.plot(history.index,history['rollingClose25'],label=ticker + ' 25 MA')
axtick.plot(history.index,history['rollingClose50'],label=ticker + ' 50 MA')

# now model it like each day, starting amount is $10,000, assume no fees on transactions
def model_growth(data,model):
    net_worth = pd.DataFrame({'net worth':np.zeros(999), 'strategy':np.zeros(999)},index=data.index[-from_mins_back:-to_mins_back])
    net_worth['strategy'] = net_worth['strategy'].astype(str)
    funds = 0.00
    price = data['Close'][-from_mins_back-1] # day to purchase
    funds -= price # purchase one share at the beginning
    shares = 1
    for day,i in enumerate(range(-from_mins_back,-to_mins_back)):
        d = data[:i]
        what_to_do = model(d)
        if what_to_do == 'buy':
            price = d['Close'][-to_mins_back]
            funds -= price
            shares += 1
        elif what_to_do == 'sell':
            price = d['Close'][-to_mins_back]
            funds += price
            shares -= 1
        elif what_to_do == 'hold':
            price = d['Close'][-to_mins_back]
            pass
        net_worth.loc[net_worth.index[day],'net worth'] = funds + price*shares
        net_worth.loc[net_worth.index[day],'strategy'] = what_to_do
    # sell at the end
    print('selling shares at the end')
    price = data['Close'][-to_mins_back]
    funds += price*shares
    shares = 0
    print(funds,shares,price)
    return net_worth

# buy and hold strategy
net_worth_BH = model_growth(history,to_buy_and_hold)
net_worth_MA3 = model_growth(history,lambda x: to_buy_hold_sell_MA(x,period='3'))
net_worth_MA5 = model_growth(history,lambda x: to_buy_hold_sell_MA(x,period='5'))
net_worth_MA10 = model_growth(history,lambda x: to_buy_hold_sell_MA(x,period='10'))
net_worth_MA15 = model_growth(history,lambda x: to_buy_hold_sell_MA(x,period='15'))
net_worth_MA25 = model_growth(history,lambda x: to_buy_hold_sell_MA(x,period='25'))
net_worth_MA50 = model_growth(history,lambda x: to_buy_hold_sell_MA(x,period='50'))

# plot strategies net worth
fig,ax = plt.subplots(figsize=(4.75,3),dpi=200,tight_layout=True)
ax.plot(net_worth_BH['net worth'],label='BH')
ax.plot(net_worth_MA3['net worth'],label='MA3')
ax.plot(net_worth_MA5['net worth'],label='MA5')
ax.plot(net_worth_MA10['net worth'],label='MA10')
ax.plot(net_worth_MA15['net worth'],label='MA15')
ax.plot(net_worth_MA25['net worth'],label='MA25')
ax.plot(net_worth_MA50['net worth'],label='MA50')
def plot_buy_and_sell(net_worth,ax,column='net worth'):
    for i in range(-from_mins_back+1,0):
        if net_worth['strategy'][i] == 'buy':
            ax.plot(net_worth.index[i],net_worth[column][i],'g^')
        elif net_worth['strategy'][i] == 'sell':
            ax.plot(net_worth.index[i],net_worth[column][i],'rv')
#plot_buy_and_sell(net_worth_BH,ax)
#plot_buy_and_sell(net_worth_MA,ax)
net_worth_MA3['Close'] = history['Close'][-from_mins_back:-to_mins_back]
net_worth_MA3['rollingClose3'] = history['rollingClose3'][-from_mins_back:-to_mins_back]
net_worth_MA5['Close'] = history['Close'][-from_mins_back:-to_mins_back]
net_worth_MA5['rollingClose5'] = history['rollingClose5'][-from_mins_back:-to_mins_back]
net_worth_MA10['Close'] = history['Close'][-from_mins_back:-to_mins_back]
net_worth_MA10['rollingClose10'] = history['rollingClose10'][-from_mins_back:-to_mins_back]
net_worth_MA15['Close'] = history['Close'][-from_mins_back:-to_mins_back]
net_worth_MA15['rollingClose15'] = history['rollingClose15'][-from_mins_back:-to_mins_back]
net_worth_MA25['Close'] = history['Close'][-from_mins_back:-to_mins_back]
net_worth_MA25['rollingClose25'] = history['rollingClose25'][-from_mins_back:-to_mins_back]
net_worth_MA50['Close'] = history['Close'][-from_mins_back:-to_mins_back]
net_worth_MA50['rollingClose50'] = history['rollingClose50'][-from_mins_back:-to_mins_back]
#plot_buy_and_sell(net_worth_MA,axtick,column='Close')
plot_buy_and_sell(net_worth_MA3,axtick,column='rollingClose3')
plot_buy_and_sell(net_worth_MA5,axtick,column='rollingClose5')
plot_buy_and_sell(net_worth_MA10,axtick,column='rollingClose10')
plot_buy_and_sell(net_worth_MA15,axtick,column='rollingClose15')
plot_buy_and_sell(net_worth_MA25,axtick,column='rollingClose25')
plot_buy_and_sell(net_worth_MA50,axtick,column='rollingClose50')
#ax.set_xlabel('days')
ax.set_ylabel('net worth $')
for tick in ax.get_xticklabels():
    tick.set_rotation(45)
ax.legend(loc='best',numpoints=1)
axtick.legend(loc='best',numpoints=1)
figtick.show()
fig.show()







days_avg = 14 # how many hours to look back to predict the future

input('Enter to exit')
