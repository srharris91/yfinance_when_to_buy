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
normalize = True # do you want to normalize the input data?
history = yf.Ticker(ticker).history(start=dstart,interval='1d',actions=False) # should give 280 rows of data (40 work days, 7 hours a day)
# label best time to buy/hold/sell if local minimum or maximum
#print(history)
labels = pd.DataFrame(history[['Close']])
ratio = 0.99 # ratio to be larger/smaller than the previous and ending day to be classified as a buy or sell
#ratio = 1.00 # ratio to be larger/smaller than the previous and ending day to be classified as a buy or sell
for i,closei in enumerate(labels['Close']):
    if i>0:
        d = labels[i-1:i+2]['Close']
        argmax = np.argmax(d)
        argmin = np.argmin(d)
        if argmin == 1:
            #what_to_do = 'buy'
            if (d[1]<=d[0]*ratio) and (d[1]<=d[0]*ratio):
                labels.loc[labels.index[i],'class'] = 0
            else:
                labels.loc[labels.index[i],'class'] = 1
        elif (argmin!=1) and (argmax != 1):
            #what_to_do = 'hold'
            labels.loc[labels.index[i],'class'] = 1
        elif argmax == 1:
            #what_to_do = 'sell'
            if (d[1]*ratio>=d[0]) and (d[1]*ratio>=d[0]):
                labels.loc[labels.index[i],'class'] = 2
            else:
                labels.loc[labels.index[i],'class'] = 1
    else:
        labels.loc[labels.index[i],'class'] = 1 # hold if first day

test_days = 180
perm = np.random.permutation(history.shape[0]-test_days-20)+20
#print(history.shape,history.loc[history.index[[2,3,4]]])
#history_train = pd.DataFrame(history.loc[history.index[perm[:-100]]])
#history_val = pd.DataFrame(history.loc[history.index[perm[-100:]]])
#print(history_train)


stock = StockDataFrame.retype(history)
image_list = ['rsi_6','rsi_12','wr_10','wr_6','macd','close_6_ema','close_6_sma','trix','tema','cci','cci_20','kdjk','dma','pdi','mdi','dx','adx','adxr','cr','tr','atr','boll']
#image_list = ['macd','tema','dma','adx','boll']
#image_list = ['rsi_6','rsi_12','wr_10','wr_6','macd','macdh','close_6_ema','close_20_ema','close_6_sma','close_12_sma','high_12_ema','high_6_ema','high_12_sma','high_6_sma','low_6_ema','low_12_ema','low_6_sma','low_12_sma','trix','trix_9_sma','tema','cci','cci_20','kdjk','kdjd','kdjj','dma','pdi','mdi','dx','adx','adxr','cr','tr','atr','boll','volume_-1_r','volume_-2_r','volume_-3_r','volume_-4_r','volume_-5_r','volume_-6_r','volume_-7_r','volume_-8_r','volume_-9_r','volume_-10_r','open_-1_r','open_-2_r','open_-3_r','open_-4_r','high_-1_r','high_-2_r','high_-3_r','high_-4_r','close_-1_r','close_-2_r','close_-3_r','close_-4_r','close_-5_r','close_-6_r','close_-7_r','close_-8_r','close_-9_r','close_-10_r','low_-1_r','low_-2_r','low_-3_r','low_-4_r','vr','vr_6_ema']
#image_list = ['rsi_6','wr_10','macd','close_6_ema','close_20_ema','high_20_ema','high_6_ema','low_6_ema','low_20_ema','trix','tema','cci','kdjk','dma','pdi','mdi','dx','adx','cr','tr','atr','boll','volume_delta','high_delta','close_delta','low_delta'] # H=10 works well
#image_list = ['rsi_6','wr_10','macd','close_6_ema','high_6_ema','low_6_ema','tema','dma','boll','volume_delta','high_delta','close_delta','low_delta']
#image_list = ['boll','volume_-1_d','close_-1_d','close_6_ema'] # works okay with H=10
#image_list = ['boll','volume_-1_d','close_-1_d'] # works okay with H=10
#image_list = ['boll','macd','volume_-1_d','volume_-2_d','volume_-3_d','volume_-4_d','volume_-5_d','volume_-6_d','volume_-7_d','volume_-8_d','volume_-10_d','volume_-11_d','volume_-12_d','volume_-13_d','volume_-14_d','volume_-15_d','close_-1_d','close_-2_d','close_-3_d','close_-4_d','close_-5_d','close_-6_d','close_-7_d','close_-8_d','close_-9_d','close_-10_d','close_-11_d','close_-12_d','close_-13_d','close_-14_d','close_-15_d',] # works okay with H=10
stock[image_list] # calculate statistics
print(stock)
print(stock.columns)
stock_train = pd.DataFrame(stock.loc[stock.index[perm[:-200]]])
stock_val = pd.DataFrame(stock.loc[stock.index[perm[-200:]]])
labels_train = pd.DataFrame(labels.loc[labels.index[perm[:-200]]])
labels_val = pd.DataFrame(labels.loc[labels.index[perm[-200:]]])
cols = len(stock.columns)

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
        #torch.nn.Dropout(0.5),
        #torch.nn.Dropout(0.2),
        #torch.nn.ReLU(),
        torch.nn.Hardswish(),
        #torch.nn.LeakyReLU(),
        #torch.nn.Dropout(0.7),
        #torch.nn.BatchNorm1d(H),
        torch.nn.Linear(H,H),
        #torch.nn.Dropout(0.5),
        #torch.nn.Dropout(0.2),
        torch.nn.Hardswish(),
        #torch.nn.ReLU(),
        #torch.nn.LeakyReLU(),
        #torch.nn.BatchNorm1d(H),
        #torch.nn.Linear(H,H),
        #torch.nn.Hardswish(),
        #torch.nn.Dropout(0.2),
        #torch.nn.ReLU(),
        #torch.nn.BatchNorm1d(H),
        torch.nn.Linear(H,D_out,bias=True),
        #torch.nn.Sigmoid()
        #torch.nn.Linear(D_in,D_out,bias=True),
        )

#loss_fn = torch.nn.CrossEntropyLoss(weight=torch.Tensor([1.0,2.0,1.0]).to(device=device,dtype=torch.float))
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
#optimizer = torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.9)

# send to GPU
historygpu = torch.Tensor(stock_train.values).to(device=device,dtype=torch.float)
# normalize on GPU
historygpu_val = torch.Tensor(stock_val.values).to(device=device,dtype=torch.float)
y = torch.Tensor(labels_train['class'].values).to(device=device,dtype=torch.long)
y_val = torch.Tensor(labels_val['class'].values).to(device=device,dtype=torch.long)
if normalize:
    historygpumean = historygpu.mean(dim=0)
    historygpustd = historygpu.std(dim=0)
    historygpu = (historygpu - historygpumean)/historygpustd
    historygpu_val = (historygpu_val - historygpumean)/historygpustd
#print(historygpu,y)
#print(historygpumean)
model.to(device)
best_model = copy.deepcopy(model)
best_model_val = copy.deepcopy(model)
best_loss = 10.0
best_loss_val = 10.0
# label the data
for e in range(10001):
    xin = historygpu + (torch.randn(historygpu.shape,dtype=historygpu.dtype,device=device))/20.0 # / 20.0
    y_pred = model(xin)
    #print(y,y_pred)

    loss = loss_fn(y_pred,y)
    if loss<=best_loss:
        with torch.no_grad():
            best_model = copy.deepcopy(model)
            best_loss = loss
            print('updating model with loss = ',best_loss.item())
            y_pred_val = model(historygpu_val)
            loss_val = loss_fn(y_pred_val,y_val)
            if loss_val<=best_loss_val:
                best_loss_val = loss_val
                best_model_val = copy.deepcopy(model)
                print('  updating model with loss and loss_val = ',best_loss.item(),best_loss_val.item())

    if e%1000 == 0:
        with torch.no_grad():
            y_pred_val = model(historygpu_val)
            loss_val = loss_fn(y_pred_val,y_val)
            print(e,loss.item(),loss_val.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #if e%1000 == 0:
        #if loss_val>=best_loss_val:
            #model = copy.deepcopy(best_model)
print('best loss = ',best_loss.item())
with torch.no_grad():
    best_model.cpu()
    best_model.eval()
    # calculate training data
    xin = historygpu.cpu()
    y_pred_train = best_model(xin).argmax(dim=1)
    #print(y_pred_train)
    # calculate validation data
    #stock = StockDataFrame.retype(stock_val)
    #stock[image_list] # calculate statistics
    historygpu_val = torch.Tensor(stock_val.values).to(device=devicecpu,dtype=torch.float)
    if normalize:
        historygpu_val = (historygpu_val - historygpumean.cpu())/historygpustd.cpu()
    xin = historygpu_val
    y_pred_val = best_model(xin)
    y_pred_val_argmax = y_pred_val.argmax(dim=1)
    #print(y_pred)
figtick,axtick = plt.subplots(figsize=(4.75,3),dpi=200,tight_layout=True)
axtick.plot(history.index,history['close'])
for i,day in enumerate(stock_train.index):
    if y_pred_train[i] == 0: # buy
        axtick.plot(day,stock_train.loc[day,'close'],'g^')
    elif (y_pred_train[i] != 0) and (y_pred_train[i] != 2): # hold
        pass
    elif y_pred_train[i] == 2: # sell
        axtick.plot(day,stock_train.loc[day,'close'],'rv')
#for i,day in enumerate(labels_train.index):
#    if labels_train['class'][i] == 0: # buy
#        ax.plot(day,stock_train.loc[day,'close'],'k^',ms=1)
#    elif (labels_train['class'][i] != 0) and (labels_train['class'][i] != 2): # hold
#        pass
#    elif labels_train['class'][i] == 2: # sell
#        ax.plot(day,stock_train.loc[day,'close'],'kv',ms=1)
for i,day in enumerate(stock_val.index):
    if y_pred_val_argmax[i] == 0: # buy
        axtick.plot(day,stock_val.loc[day,'close'],'g^')
    elif (y_pred_val_argmax[i] != 0) and (y_pred_val_argmax[i] != 2): # hold
        pass
    elif y_pred_val_argmax[i] == 2: # sell
        axtick.plot(day,stock_val.loc[day,'close'],'rv')
#ax.axvline(stock_train.index[-1],color='k')
#ax.axvline(stock_val.index,color='k')
figtick.show()

yval = torch.Tensor(labels_val['class'].values).to(device=devicecpu,dtype=torch.long)
print(loss_fn.cpu()(y_pred_val,yval).item())

def model_growth_of_best(data):
    num_days = data.shape[0]
    net_worth = pd.DataFrame({'net worth':np.zeros(num_days), 'strategy':np.zeros(num_days)},index=data.index)
    net_worth['strategy'] = net_worth['strategy'].astype(str)
    funds = 1000.00
    #price = data['Close'][-from_days_back-1] # day to purchase
    #funds -= price # purchase one share at the beginning
    shares = 0
    for i,day in enumerate(data.index[:-1]):
        what_to_do = data.loc[day,'class']
        price = data.loc[day,'Close']
        if what_to_do == 0: #'buy':
            num_shares_to_buy = funds//price
            funds -= num_shares_to_buy*price
            shares += num_shares_to_buy
        elif what_to_do == 2: #'sell':
            funds += shares*price
            shares = 0
        elif what_to_do == 1: #'hold':
            pass
        net_worth.loc[day,'net worth'] = funds + price*shares
        net_worth.loc[day,'strategy'] = what_to_do
    for i,day in enumerate(data.index[-1:]):
        price = data.loc[day,'Close']
        net_worth.loc[day,'net worth'] = funds + price*shares
        net_worth.loc[day,'strategy'] = 'hold'
    # sell at the end
    #print('selling shares at the end')
    price = data['Close'][-1]
    funds += price*shares
    shares = 0
    print(' net funds at the end of best = ',funds)
    return net_worth

def model_growth_of_buy_and_hold(data):
    num_days = data.shape[0]
    net_worth = pd.DataFrame({'net worth':np.zeros(num_days), 'strategy':np.zeros(num_days)},index=data.index)
    net_worth['strategy'] = net_worth['strategy'].astype(str)
    funds = 1000.00
    #price = data['Close'][-from_days_back-1] # day to purchase
    #funds -= price # purchase one share at the beginning
    shares = 0
    for i,day in enumerate(data.index[:-1]):
        price = data.loc[day,'Close']
        if i==0:
            what_to_do = 0
        else:
            what_to_do = 1
        if what_to_do == 0: #'buy':
            num_shares_to_buy = funds//price
            funds -= num_shares_to_buy*price
            shares += num_shares_to_buy
        elif what_to_do == 2: #'sell':
            funds += shares*price
            shares = 0
        elif what_to_do == 1: #'hold':
            pass
        net_worth.loc[day,'net worth'] = funds + price*shares
        net_worth.loc[day,'strategy'] = what_to_do
    for i,day in enumerate(data.index[-1:]):
        price = data.loc[day,'Close']
        net_worth.loc[day,'net worth'] = funds + price*shares
        net_worth.loc[day,'strategy'] = 'hold'
    # sell at the end
    price = data['Close'][-1]
    funds += price*shares
    shares = 0
    print(' net funds at the end of BnH = ',funds)
    return net_worth
def model_growth_of_NN(data):
    num_days = data.shape[0]
    net_worth = pd.DataFrame({'net worth':np.zeros(num_days), 'strategy':np.zeros(num_days)},index=data.index)
    net_worth['strategy'] = net_worth['strategy'].astype(str)
    funds = 1000.00
    #price = data['Close'][-from_days_back-1] # day to purchase
    #funds -= price # purchase one share at the beginning
    shares = 0
    for i,day in enumerate(data.index):
        what_to_do = data.loc[day,'predicted']
        price = data.loc[day,'close']
        if what_to_do == 0: #'buy':
            num_shares_to_buy = funds//price
            funds -= num_shares_to_buy*price
            shares += num_shares_to_buy
        elif what_to_do == 2: #'sell':
            funds += shares*price
            shares = 0
        elif what_to_do == 1: #'hold':
            pass
        net_worth.loc[day,'net worth'] = funds + price*shares
        net_worth.loc[day,'strategy'] = what_to_do
    # sell at the end
    price = data['close'][-1]
    funds += price*shares
    shares = 0
    print(' net funds at the end = ',funds)
    return net_worth

# buy and hold strategy
net_worth_best = model_growth_of_best(labels[-100:])
net_worth_buy_and_hold = model_growth_of_buy_and_hold(labels[-100:])
labels_test = stock.loc[stock.index[-100:]]
with torch.no_grad():
    best_model.cpu()
    best_model.eval()
    best_model_val.cpu()
    best_model_val.eval()
    historygpu_test = torch.Tensor(labels_test.values).to(device=devicecpu,dtype=torch.float)
    if normalize:
        historygpu_test = (historygpu_test - historygpumean.cpu())/historygpustd.cpu()
    xin = historygpu_test
    y_pred_test = best_model(xin)
    y_pred_test_argmax = y_pred_test.argmax(dim=1)
    y_pred_test_val = best_model_val(xin)
    y_pred_test_val_argmax = y_pred_test_val.argmax(dim=1)
    labels_test['predicted'] = y_pred_test_argmax.detach().clone()
axtick.axvline(labels_test.index[0],color='k')
print('------------ best model ----------------')
for i,day in enumerate(labels_test.index):
    if labels_test.loc[day,'predicted'] == 0: # buy
        #print(day,'buy ',y_pred_test[i].numpy(),' vals, or sigmoid(val) = ',torch.sigmoid(y_pred_test[i]).numpy())
        axtick.plot(day,labels_test.loc[day,'close'],'g^')
        axtick.text(day,labels_test.loc[day,'close'],'{:.2f},{:.2f},{:.2f}'.format(*torch.sigmoid(y_pred_test[i])))
    elif (labels_test.loc[day,'predicted'] != 0) and (labels_test.loc[day,'predicted'] != 2): # hold
        #print(day,'hold',y_pred_test[i].numpy(),' vals, or sigmoid(val) = ',torch.sigmoid(y_pred_test[i]).numpy())
        pass
    elif labels_test.loc[day,'predicted'] == 2: # sell
        axtick.plot(day,labels_test.loc[day,'close'],'rv')
        axtick.text(day,labels_test.loc[day,'close'],'{:.2f},{:.2f},{:.2f}'.format(*torch.sigmoid(y_pred_test[i])))
        #print(day,'sell',y_pred_test[i].numpy(),' vals, or sigmoid(val) = ',torch.sigmoid(y_pred_test[i]).numpy())
net_worth_NN = model_growth_of_NN(labels_test)
with torch.no_grad():
    labels_test['predicted'] = y_pred_test_val_argmax.detach().clone()
net_worth_val_NN = model_growth_of_NN(labels_test)

fig,ax = plt.subplots(figsize=(4.75,3),dpi=200,tight_layout=True)
ax.plot(net_worth_best['net worth'],label='best')
ax.plot(net_worth_buy_and_hold['net worth'],label='BnH')
ax.plot(net_worth_NN['net worth'],label='NN')
ax.plot(net_worth_val_NN['net worth'],label='NN val')
ax.set_xlabel('days')
ax.set_ylabel('net worth $')
for tick in ax.get_xticklabels():
    tick.set_rotation(45)
ax.legend(loc='best',numpoints=1)
fig.show()

figtick,axtick = plt.subplots(figsize=(4.75,3),dpi=200,tight_layout=True)
figtick.suptitle('best validation model')
with torch.no_grad():
    labels_test['predicted'] = y_pred_test_val_argmax.detach().clone()
axtick.plot(labels_test.index,labels_test['close'])
print('------------ best validation model ----------------')
for i,day in enumerate(labels_test.index):
    if labels_test.loc[day,'predicted'] == 0: # buy
        axtick.plot(day,labels_test.loc[day,'close'],'g^')
        #print(day,'buy ',y_pred_test_val[i].numpy(),torch.sigmoid(y_pred_test_val[i]).numpy())
        axtick.text(day,labels_test.loc[day,'close'],'{:.2f},{:.2f},{:.2f}'.format(*torch.sigmoid(y_pred_test_val[i])))
    elif (labels_test.loc[day,'predicted'] != 0) and (labels_test.loc[day,'predicted'] != 2): # hold
        axtick.text(day,labels_test.loc[day,'close'],'{:.2f},{:.2f},{:.2f}'.format(*torch.sigmoid(y_pred_test_val[i])))
        pass
        #print(day,'hold',y_pred_test_val[i].numpy(),torch.sigmoid(y_pred_test_val[i]).numpy())
    elif labels_test.loc[day,'predicted'] == 2: # sell
        axtick.plot(day,labels_test.loc[day,'close'],'rv')
        #print(day,'sell',y_pred_test_val[i].numpy(),torch.sigmoid(y_pred_test_val[i]).numpy())
        axtick.text(day,labels_test.loc[day,'close'],'{:.2f},{:.2f},{:.2f}'.format(*torch.sigmoid(y_pred_test_val[i])))
figtick.show()

figtick,axtick = plt.subplots(figsize=(4.75,3),dpi=200,tight_layout=True)
figtick.suptitle('best training model')
with torch.no_grad():
    labels_test['predicted'] = y_pred_test_argmax.detach().clone()
axtick.plot(labels_test.index,labels_test['close'])
for i,day in enumerate(labels_test.index):
    if labels_test.loc[day,'predicted'] == 0: # buy
        axtick.plot(day,labels_test.loc[day,'close'],'g^')
        #print(day,'buy ',y_pred_test[i].numpy(),torch.sigmoid(y_pred_test[i]).numpy())
        axtick.text(day,labels_test.loc[day,'close'],'{:.2f},{:.2f},{:.2f}'.format(*torch.sigmoid(y_pred_test[i])))
    elif (labels_test.loc[day,'predicted'] != 0) and (labels_test.loc[day,'predicted'] != 2): # hold
        pass
    elif labels_test.loc[day,'predicted'] == 2: # sell
        axtick.plot(day,labels_test.loc[day,'close'],'rv')
        #print(day,'sell',y_pred_test[i].numpy(),torch.sigmoid(y_pred_test[i]).numpy())
        axtick.text(day,labels_test.loc[day,'close'],'{:.2f},{:.2f},{:.2f}'.format(*torch.sigmoid(y_pred_test[i])))
figtick.show()

#test = (y_pred_test_argmax == y_pred_test_val_argmax)
#print(test.shape,test.sum())
#print(test)
with torch.no_grad():
    labels_test['predicted'] = y_pred_test_argmax.detach().clone()
print('best trained model says ',labels_test.index[-1],labels_test['close'][-1],labels_test['predicted'][-1])
with torch.no_grad():
    labels_test['predicted'] = y_pred_test_val_argmax.detach().clone()
print('best validation model says ',labels_test.index[-1],labels_test['close'][-1],labels_test['predicted'][-1])

input('Enter to exit')
