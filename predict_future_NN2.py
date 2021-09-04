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
dstart = (today-timedelta(days=10000)).isoformat() # 60 days prior to today
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

perm = np.random.permutation(history.shape[0]-21)+20
#print(history.shape,history.loc[history.index[[2,3,4]]])
#history_train = pd.DataFrame(history.loc[history.index[perm[:-100]]])
#history_val = pd.DataFrame(history.loc[history.index[perm[-100:]]])
#print(history_train)


stock = StockDataFrame.retype(history)
#image_list = ['rsi_6','rsi_12','wr_10','wr_6','macd','close_6_ema','close_6_sma','trix','tema','cci','cci_20','kdjk','dma','pdi','mdi','dx','adx','adxr','cr','tr','atr','boll']
#image_list = ['macd','tema','dma','adx','boll']
#image_list = ['rsi_6','rsi_12','wr_10','wr_6','macd','macdh','close_6_ema','close_20_ema','close_6_sma','close_12_sma','high_12_ema','high_6_ema','high_12_sma','high_6_sma','low_6_ema','low_12_ema','low_6_sma','low_12_sma','trix','trix_9_sma','tema','cci','cci_20','kdjk','kdjd','kdjj','dma','pdi','mdi','dx','adx','adxr','cr','tr','atr','boll','volume_delta','volume_-2_d','volume_-3_d','volume_-4_d','open_delta','open_-2_d','open_-3_d','open_-4_d','high_delta','high_-2_d','high_-3_d','high_-4_d','close_delta','close_-2_d','close_-3_d','close_-4_d','low_delta','low_-2_d','low_-3_d','low_-4_d','close_-1_r','close_-2_r','close_-3_r','vr','vr_6_ema']
#image_list = ['rsi_6','wr_10','macd','close_6_ema','close_20_ema','high_20_ema','high_6_ema','low_6_ema','low_20_ema','trix','tema','cci','kdjk','dma','pdi','mdi','dx','adx','cr','tr','atr','boll','volume_delta','high_delta','close_delta','low_delta'] # H=10 works well
#image_list = ['rsi_6','wr_10','macd','close_6_ema','high_6_ema','low_6_ema','tema','dma','boll','volume_delta','high_delta','close_delta','low_delta']
#image_list = ['boll','volume_-1_d','close_-1_d','close_6_ema'] # works okay with H=10
image_list = ['boll','volume_-1_d','close_-1_d'] # works okay with H=10
stock[image_list] # calculate statistics
print(stock)
print(stock.columns)
stock_train = pd.DataFrame(stock.loc[stock.index[perm[:-100]]])
stock_val = pd.DataFrame(stock.loc[stock.index[perm[-100:]]])
labels_train = pd.DataFrame(labels.loc[labels.index[perm[:-100]]])
labels_val = pd.DataFrame(labels.loc[labels.index[perm[-100:]]])
cols = len(stock.columns)

# we want to predict the decision to buy, hold, or sell on the current day to maximize profits
D_in = cols
H = 10 # hidden layers
D_out = 3 # predict buy, hold, sell classes

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
devicecpu = torch.device('cpu')
print('device = ',device)
model = torch.nn.Sequential(
        #torch.nn.BatchNorm1d(D_in),
        torch.nn.Linear(D_in,H),
        #torch.nn.Dropout(0.2),
        #torch.nn.ReLU(),
        torch.nn.Hardswish(),
        #torch.nn.LeakyReLU(),
        #torch.nn.BatchNorm1d(H),
        #torch.nn.Linear(H,H),
        #torch.nn.Dropout(0.2),
        #torch.nn.ReLU(),
        #torch.nn.LeakyReLU(),
        #torch.nn.BatchNorm1d(H),
        #torch.nn.Linear(H,H),
        #torch.nn.Dropout(0.2),
        #torch.nn.ReLU(),
        #torch.nn.BatchNorm1d(H),
        torch.nn.Linear(H,D_out,bias=False),
        #torch.nn.Sigmoid()
        #torch.nn.Linear(D_in,D_out,bias=True),
        )

loss_fn = torch.nn.CrossEntropyLoss(weight=torch.Tensor([1.0,1.0,1.0]).to(device=device,dtype=torch.float))
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
#optimizer = torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.9)

# send to GPU
historygpu = torch.Tensor(stock_train.values).to(device=device,dtype=torch.float)
# normalize on GPU
historygpumean = historygpu.mean(dim=0)
historygpustd = historygpu.std(dim=0)
historygpu = (historygpu - historygpumean)/historygpustd
y = torch.Tensor(labels_train['class'].values).to(device=device,dtype=torch.long)
#print(historygpu,y)
#print(historygpumean)
model.to(device)
best_model = copy.deepcopy(model)
best_loss = 1.0
# label the data
for e in range(30001):
    xin = historygpu #+ (torch.randn(historygpu.shape,dtype=historygpu.dtype,device=device))/20.0 # / 20.0
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
    #stock = StockDataFrame.retype(stock_val)
    #stock[image_list] # calculate statistics
    historygpu_val = torch.Tensor(stock_val.values).to(device=devicecpu,dtype=torch.float)
    historygpu_val = (historygpu_val - historygpumean.cpu())/historygpustd.cpu()
    xin = historygpu_val
    y_pred_val = best_model(xin)
    y_pred_val_argmax = y_pred_val.argmax(dim=1)
    #print(y_pred)
fig,ax = plt.subplots(figsize=(4.75,3),dpi=200,tight_layout=True)
ax.plot(history.index,history['close'])
for i,day in enumerate(stock_train.index):
    if y_pred_train[i] == 0: # buy
        ax.plot(day,stock_train.loc[day,'close'],'g^')
    elif (y_pred_train[i] != 0) and (y_pred_train[i] != 2): # hold
        pass
    elif y_pred_train[i] == 2: # sell
        ax.plot(day,stock_train.loc[day,'close'],'rv')
#for i,day in enumerate(labels_train.index):
#    if labels_train['class'][i] == 0: # buy
#        ax.plot(day,stock_train.loc[day,'close'],'k^',ms=1)
#    elif (labels_train['class'][i] != 0) and (labels_train['class'][i] != 2): # hold
#        pass
#    elif labels_train['class'][i] == 2: # sell
#        ax.plot(day,stock_train.loc[day,'close'],'kv',ms=1)
for i,day in enumerate(stock_val.index):
    if y_pred_val_argmax[i] == 0: # buy
        ax.plot(day,stock_val.loc[day,'close'],'g^')
    elif (y_pred_val_argmax[i] != 0) and (y_pred_val_argmax[i] != 2): # hold
        pass
    elif y_pred_val_argmax[i] == 2: # sell
        ax.plot(day,stock_val.loc[day,'close'],'rv')
#ax.axvline(stock_train.index[-1],color='k')
#ax.axvline(stock_val.index,color='k')
fig.show()

yval = torch.Tensor(labels_val['class'].values).to(device=devicecpu,dtype=torch.long)
print(loss_fn.cpu()(y_pred_val,yval).item())

input('Enter to exit')
