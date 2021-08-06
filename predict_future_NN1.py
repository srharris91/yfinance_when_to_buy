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
dstart = (today-timedelta(days=60)).isoformat() # 60 days prior to today
#print(dstart,dend)

ticker = 'TQQQ'
history = yf.Ticker(ticker).history(start=dstart,interval='1h') # should give 280 rows of data (40 work days, 7 hours a day)
#print(history.index[-4])
#print(history.index[-3])
#print(history.index[-2])
#print(history.index[-1])

rows,cols = history.shape
hours_back = 10 # how many hours to look back to predict the future
N = 200 # how many samples
cols = 2 # always 7 columns ['Open','High','Low','Close','Volume','Dividends','Stock Splits'] # unless you only want Close and Volume
# we want to predict the next hour 'Close' price
D_in = hours_back*cols
H = 100 # hidden layers
D_out = 2 # predict 'Close' and 'Volume price at next hour

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
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
        #torch.nn.Linear(D_in,D_out,bias=True),
        )

loss_fn = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

# send to GPU
historygpu = torch.Tensor(history[['Close','Volume']].values).to(device=device,dtype=torch.float)
# normalize on GPU
historygpumean = historygpu.mean(dim=0)
historygpustd = historygpu.std(dim=0)
historygpu = (historygpu - historygpumean)/historygpustd
print(historygpumean)
model.to(device)
best_model = copy.deepcopy(model)
best_loss = 1.0
for e in range(100):
    for t in range(rows-N-hours_back-4):
        xin = torch.stack([historygpu[ti:ti+hours_back].flatten() for ti in range(t,t+N)])#).to(device=device,dtype=torch.float)
        xin += (torch.randn(xin.shape,dtype=xin.dtype,device=device))/20.0
        #y = torch.stack([historygpu[ti+hours_back,3] for ti in range(t,t+N)])[:,None]#).to(device=device,dtype=torch.float)
        y = torch.stack([historygpu[ti+hours_back,:] for ti in range(t,t+N)])#).to(device=device,dtype=torch.float)
        #print(xin.shape,y.shape,xin[0,3::7],y[0])
        y_pred = model(xin)
        #print(y,y_pred)

        loss = loss_fn(y_pred,y)
        if loss<=best_loss:
            best_loss = loss
            best_model = copy.deepcopy(model)

        if t%10 == 0:
            print(e,loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
print('best loss = ',best_loss)
with torch.no_grad():
    best_model.cpu()
    best_model.eval()
    fig,ax = plt.subplots(figsize=(4.75,3),dpi=200,tight_layout=True)
    xin = torch.from_numpy(np.stack([history[['Close','Volume']][ti:ti+hours_back].values for ti in range(0,rows-hours_back,hours_back)])).to(device='cpu',dtype=torch.float)
    xin = (xin-historygpumean.cpu()[None,:])/historygpustd.cpu()[None,:] # normalized data
    xin = xin.reshape((xin.shape[0],-1))
    y_pred = best_model(xin)
    y_pred = y_pred*historygpustd.cpu()+historygpumean.cpu() # un-normalize prediction
    #print(y_pred)
    ax.plot(history.index,history['Close'],label=ticker)
    ax.plot(history.index[1:rows-hours_back+1:hours_back],y_pred[:,0],'.',label=ticker+' model')
    # predict what it will be this hour
    xin = torch.from_numpy(np.stack([history[['Close','Volume']][-hours_back-1:-1].values])).to(device='cpu',dtype=torch.float)
    xin = (xin-historygpumean.cpu()[None,:])/historygpustd.cpu()[None,:] # normalized data
    xin = xin.reshape((xin.shape[0],-1))
    y_pred = best_model(xin)
    y_pred = y_pred*historygpustd.cpu()+historygpumean.cpu() # un-normalize prediction
    ax.plot(history.index[-1],y_pred[:,0],'.',label=ticker+' pred for this hour')
    #print('most current time ',history.index[-1],' with price ',history['Close'][-1])
    print(ticker + ' price is currently       $',history['Close'][-1],' this hour ',history.index[-1])
    #print(type(history.index[-1]))
    ax.set_ylabel(ticker+' price $')
    #ax.set_xticklabels(ax.get_xticks(),rotation=45)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    # predict what it'll be next hour
    xin = torch.from_numpy(np.stack([history[['Close','Volume']][-hours_back:].values])).to(device='cpu',dtype=torch.float)
    xin = (xin-historygpumean.cpu()[None,:])/historygpustd.cpu()[None,:] # normalized data
    xin = xin.reshape((xin.shape[0],-1))
    y_pred = best_model(xin)
    y_pred = y_pred*historygpustd.cpu()+historygpumean.cpu() # un-normalize prediction
    print(ticker + ' price is predicted to be $',y_pred[-1,0].item(),' next hour ',history.index[-1]+pd.to_timedelta(1,unit='h'))
    ax.plot(history.index[-1]+pd.to_timedelta(1,unit='h'),y_pred[:,0],'.',label=ticker+' pred for next hour')
    ax.legend(loc='best',numpoints=1)
    fig.show()

    input('Enter to exit')
