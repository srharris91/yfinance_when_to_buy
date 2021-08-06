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
dstart = (today-timedelta(days=59)).isoformat() # 60 days prior to today
#print(dstart,dend)

ticker = 'TQQQ'
history = yf.Ticker(ticker).history(start=dstart,interval='15m') # should give 280 rows of data (40 work days, 7 hours a day)
#print(history.index[-4])
#print(history.index[-3])
#print(history.index[-2])
#print(history.index[-1])

rows,cols = history.shape
print('rows = ',rows)
hours_back = 20 # how many hours to look back to predict the future
N = 900 # how many samples
cols = 2 # always 7 columns ['Open','High','Low','Close','Volume','Dividends','Stock Splits'] # unless you only want Close and Volume
# we want to predict the next hour 'Close' and 'Volume'
D_in = cols
H = 10 # hidden layers
D_out = 2 # predict 'Close' and 'Volume price at next hour

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
print('device = ',device)
class LSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Readout layer
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim,dtype=x.dtype,device=x.device).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim,dtype=x.dtype,device=x.device).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        #print(x.shape,h0.shape,c0.shape)
        #print(self.lstm(x, (h0.detach(), c0.detach())))
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 32, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        out = self.fc(out[:, -1, :]) 
        # out.size() --> 100, 10
        return out
model = LSTM(D_in,H,3,D_out)
print(model)

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
    for t in range(rows-N-hours_back):
        xin = torch.stack([historygpu[ti:ti+hours_back] for ti in range(t,t+N)])#).to(device=device,dtype=torch.float)
        #print(xin.shape,torch.stack([historygpu[ti:ti+hours_back] for ti in range(t,t+N)]).shape)
        xin += (torch.randn(xin.shape,dtype=xin.dtype,device=device))/20.0
        #y = torch.stack([historygpu[ti+hours_back,3] for ti in range(t,t+N)])[:,None]#).to(device=device,dtype=torch.float)
        y = torch.stack([historygpu[ti+hours_back,:] for ti in range(t,t+N)])#).to(device=device,dtype=torch.float)
        #print(xin.shape,y.shape,xin[0,3::7],y[0])
        y_pred = model(xin)
        #print(y.shape,y_pred.shape)
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
    #print(xin.shape)
    #xin = xin.reshape((xin.shape[0],-1))
    #print(xin.shape)
    y_pred = best_model(xin)
    y_pred = y_pred*historygpustd.cpu()+historygpumean.cpu() # un-normalize prediction
    #print(y_pred)
    ax.plot(history.index,history['Close'],label=ticker)
    ax.plot(history.index[1:rows-hours_back+1:hours_back],y_pred[:,0],'.',label=ticker+' model')
    # predict what it will be this hour
    xin = torch.from_numpy(np.stack([history[['Close','Volume']][-hours_back-1:-1].values])).to(device='cpu',dtype=torch.float)
    xin = (xin-historygpumean.cpu()[None,:])/historygpustd.cpu()[None,:] # normalized data
    #xin = xin.reshape((xin.shape[0],-1))
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
    #xin = xin.reshape((xin.shape[0],-1))
    y_pred = best_model(xin)
    y_pred = y_pred*historygpustd.cpu()+historygpumean.cpu() # un-normalize prediction
    print(ticker + ' price is predicted to be $',y_pred[-1,0].item(),' next hour ',history.index[-1]+pd.to_timedelta(1,unit='h'))
    ax.plot(history.index[-1]+pd.to_timedelta(1,unit='h'),y_pred[:,0],'.',label=ticker+' pred for next hour')
    ax.legend(loc='best',numpoints=1)
    fig.show()

    input('Enter to exit')
