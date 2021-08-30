import torch
import yfinance as yf
import numpy as np
import copy

from datetime import date,timedelta
import pandas as pd

from stockstats import StockDataFrame

import matplotlib.pyplot as plt
plt.style.use('seaborn-notebook')

today = date.today()
dend = today.isoformat()
dstart = (today-timedelta(days=2000)).isoformat() # 60 days prior to today
#print(dstart,dend)

ticker = 'TQQQ'
history = yf.Ticker(ticker).history(start=dstart,interval='1d',actions=False) # should give 280 rows of data (40 work days, 7 hours a day)
# label best time to buy/hold/sell if local minimum or maximum
#print(history)
labels = pd.DataFrame(history[['Close']])
for i,closei in enumerate(labels['Close']):
    if i>0:
        argmax = np.argmax(labels[i-1:i+2]['Close'])
        argmin = np.argmin(labels[i-1:i+2]['Close'])
        if argmin == 1:
            #what_to_do = 'buy'
            labels.loc[labels.index[i],'class'] = 0
        elif (argmin!=1) and (argmax != 1):
            #what_to_do = 'hold'
            labels.loc[labels.index[i],'class'] = 1
        elif argmax == 1:
            #what_to_do = 'sell'
            labels.loc[labels.index[i],'class'] = 2
    else:
        labels.loc[labels.index[i],'class'] = 1 # hold if first day

history_train = pd.DataFrame(history[:-100])
labels_train = pd.DataFrame(labels[:-100])
history_val = pd.DataFrame(history[-100:])
labels_val = pd.DataFrame(labels[-100:])


stock = StockDataFrame.retype(history_train)
image_list = ['rsi_6','rsi_12','wr_10','wr_6','macd','close_6_ema','close_6_sma','trix','tema','cci','cci_20','kdjk','dma','pdi','mdi','dx','adx','adxr','cr','tr','atr','boll']
stock[image_list] # calculate statistics
cols = len(stock.columns)

rows = history_train.shape[0]
hours_back = 10 # how many hours to look back to predict the future
N = 200 # how many samples
# we want to predict the decision to buy, hold, or sell on the current day to maximize profits
D_in = cols
H = 100 # hidden layers
D_out = 3 # predict buy, hold, sell classes

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
devicecpu = torch.device('cpu')
print('device = ',device)
model = torch.nn.Sequential(
        #torch.nn.BatchNorm1d(D_in),
        torch.nn.Linear(D_in,H),
        torch.nn.ReLU(),
        #torch.nn.BatchNorm1d(H),
        #torch.nn.Linear(H,H),
        #torch.nn.ReLU(),
        #torch.nn.BatchNorm1d(H),
        #torch.nn.Linear(H,H),
        #torch.nn.ReLU(),
        #torch.nn.BatchNorm1d(H),
        torch.nn.Linear(H,D_out,bias=True),
        #torch.nn.Sigmoid()
        #torch.nn.Linear(D_in,D_out,bias=True),
        )

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

# send to GPU
historygpu = torch.Tensor(stock[10:].values).to(device=device,dtype=torch.float)
# normalize on GPU
historygpumean = historygpu.mean(dim=0)
historygpustd = historygpu.std(dim=0)
historygpu = (historygpu - historygpumean)/historygpustd
y = torch.Tensor(labels_train[10:]['class'].values).to(device=device,dtype=torch.long)
#print(historygpumean)
model.to(device)
best_model = copy.deepcopy(model)
best_loss = 1.0
# label the data
for e in range(10000):
    xin = historygpu + (torch.randn(historygpu.shape,dtype=historygpu.dtype,device=device))/20.0
    #y = torch.stack([historygpu[ti+hours_back,3] for ti in range(t,t+N)])[:,None]#).to(device=device,dtype=torch.float)
    #y = historygpu[ti+hours_back,:] for ti in range(t,t+N)])#).to(device=device,dtype=torch.float)
    #print(xin.shape,y.shape,xin[0,3::7],y[0])
    y_pred = model(xin)
    #print(y,y_pred)

    loss = loss_fn(y_pred,y)
    if loss<=best_loss:
        best_loss = loss
        best_model = copy.deepcopy(model)

    if e%1000 == 0:
        print(e,loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print('best loss = ',best_loss)
with torch.no_grad():
    best_model.cpu()
    best_model.eval()
    # calculate training data
    xin = historygpu.cpu()
    y_pred_train = best_model(xin).argmax(dim=1)
    print(y_pred_train)
    # calculate validation data
    stock = StockDataFrame.retype(history_val)
    stock[image_list] # calculate statistics
    historygpu_val = torch.Tensor(stock[10:].values).to(device=devicecpu,dtype=torch.float)
    historygpu_val = (historygpu_val - historygpumean.cpu())/historygpustd.cpu()
    xin = historygpu_val
    y_pred_val = best_model(xin).argmax(dim=1)
    #print(y_pred)
fig,ax = plt.subplots(figsize=(4.75,3),dpi=200,tight_layout=True)
ax.plot(history.index,history['Close'])
for i,day in enumerate(history_train.index[10:]):
    if y_pred_train[i] == 0: # buy
        ax.plot(day,history_train.loc[day,'close'],'g^')
    elif (y_pred_train[i] != 0) and (y_pred_train[i] != 2): # hold
        pass
    elif y_pred_train[i] == 2: # sell
        ax.plot(day,history_train.loc[day,'close'],'rv')
for i,day in enumerate(history_val.index[10:]):
    if y_pred_val[i] == 0: # buy
        ax.plot(day,history_val.loc[day,'close'],'g^')
    elif (y_pred_val[i] != 0) and (y_pred_val[i] != 2): # hold
        pass
    elif y_pred_val[i] == 2: # sell
        ax.plot(day,history_val.loc[day,'close'],'rv')
ax.axvline(history_train.index[-1],color='k')
ax.axvline(history_val.index[10],color='k')
fig.show()

input('Enter to exit')
