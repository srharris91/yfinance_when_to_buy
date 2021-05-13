import json
import yfinance as yf
from datetime import date,timedelta
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
plt.style.use('seaborn-notebook')

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def printBUY(string): print(bcolors.OKGREEN + string + bcolors.ENDC)
def printHOLD(string): print(bcolors.FAIL + string + bcolors.ENDC)
def printHEADER(string): print(bcolors.HEADER + string + bcolors.ENDC)

days_to_subtract = 30
period = timedelta(days=days_to_subtract)
days_to_subtract_long = 2555 #1095
period_long = timedelta(days=days_to_subtract_long)
today = date.today()
#d = today.strftime("%d/%m/%Y")
dend = today.isoformat()
dstart = (today - period).isoformat()
dstart_long = (today - period_long).isoformat()

print('dstart = ',dstart)
print('dend   = ',dend)

with open('tickers.json') as f:
    data = json.load(f)
tickers = data['tickers']

#printHEADER('----------------------------------------------')

frmts = '{:^s}'
frmts2 = '{: <8s}'
cols = [
        frmts.format(bcolors.OKCYAN + str(dend) + bcolors.ENDC),
        frmts.format(bcolors.OKCYAN + '52 high' + bcolors.ENDC),
        frmts.format(bcolors.OKCYAN + '52 low' + bcolors.ENDC),
        frmts.format(bcolors.OKCYAN + '52 0.95' + bcolors.ENDC),
        frmts.format(bcolors.OKCYAN + '52 0.90' + bcolors.ENDC),
        frmts.format(bcolors.OKCYAN + '52 0.85' + bcolors.ENDC),
        frmts.format(bcolors.OKCYAN + '50 day avg' + bcolors.ENDC),
        frmts.format(bcolors.OKCYAN + '200 day avg' + bcolors.ENDC),
        frmts.format(bcolors.OKCYAN + str(dstart) + bcolors.ENDC),
        ]
frmt = '{:g}'
to_print = pd.DataFrame(columns=cols)
fig,ax = plt.subplots(tight_layout=True)
fig2,ax2 = plt.subplots(tight_layout=True)
ax.set_ylabel(r'$\frac{\mathrm{val}}{\mathrm{val(0)}}$')
ax2.set_ylabel(r'$\frac{\mathrm{val}}{\mathrm{val(0)}}$')
#print(to_print)
for i,ti in enumerate(tickers):
    #try:
        tick = yf.Ticker(ti)
        history = tick.history(start=dstart,end=dend)
        history_long = tick.history(start=dstart_long,end=dend)
        history_long_short_rolling_mean = history_long['Close'].rolling(5).mean()
        history_long_short_rolling_mean2 = history_long['Close'].rolling(50).mean()
        history_long_long_rolling_mean = history_long['Close'].rolling(200).mean()
        ax.plot_date(history_long.index,history_long['Close']/history_long['Close'][0],'.-',color='C'+str(i),label=ti)
        ax.plot_date(history_long.index,history_long_short_rolling_mean/history_long['Close'][0],':',color='C'+str(i),label=ti+' rolling mean of 5 days')
        ax.plot_date(history_long.index,history_long_short_rolling_mean2/history_long['Close'][0],'-.',color='C'+str(i),label=ti+' rolling mean of 50 days')
        ax.plot_date(history_long.index,history_long_long_rolling_mean/history_long['Close'][0],'--',color='C'+str(i),label=ti+' rolling mean of 200 days')
        ax2.plot_date(history.index,history['Close']/history['Close'][0],'.-',color='C'+str(i),label=ti)
        fifty2High = tick.info['fiftyTwoWeekHigh']
        fifty2Low  = tick.info['fiftyTwoWeekLow']
        close  = tick.info['previousClose']
        firstHistory = history['Close'][0]
        fifty2Test = (fifty2High - 0.05*(fifty2High-fifty2Low))
        fifty2Test2 = (fifty2High - 0.10*(fifty2High-fifty2Low))
        fifty2Test3 = (fifty2High - 0.15*(fifty2High-fifty2Low))
        fiftyDayAverage = tick.info['fiftyDayAverage']
        twoHundredDayAverage = tick.info['twoHundredDayAverage']
        # tests to buy or hold
        # test if 0.9 of 52 week high
        #printHEADER(ti)
        printHEADER(frmts2.format(ti) + tick.info['longName'])
        #to_printi = pd.DataFrame(
                #[[
                    #dend:close,
                    #'52 week high':fifty2High,
                    #'52 week low':fifty2Low,
                    #'52 week test 0.95':fifty2Test,
                    #'52 week test 0.90':fifty2Test2,
                    #'52 week test 0.85':fifty2Test3,
                    #'50 day average':fiftyDayAverage,
                    #'200 day average':twoHundredDayAverage,
                    #dstart:firstHistory 
                    #str(close),
                    #str(fifty2High),
                    #str(fifty2Low),
                    #str(fifty2Test),
                    #str(fifty2Test2),
                    #str(fifty2Test3),
                    #str(fiftyDayAverage),
                    #str(twoHundredDayAverage),
                    #str(firstHistory),
                    #]],
                #columns=cols,
                #index=[ti,])
        #to_print.append(to_printi)
        #print(to_print)
        #display(to_print)
        #print(to_print.to_string())
        #display(to_print.style.bar(subset=['200 day average',],color='#d65f5f').render())
        tis = ti #frmts.format(bcolors.HEADER + ti + bcolors.ENDC)
        to_print.loc[tis,cols[0]] = frmts.format(bcolors.HEADER + frmt.format(close) + bcolors.ENDC)
        to_print.loc[tis,cols[1]] = frmts.format(bcolors.HEADER + frmt.format(fifty2High) + bcolors.ENDC)
        to_print.loc[tis,cols[2]] = frmts.format(bcolors.HEADER + frmt.format(fifty2Low) + bcolors.ENDC)
        if close<=fifty2Test:
            #printBUY(' buy based on 0.95 of 52 week high')
            to_print.loc[tis,cols[3]] = frmts.format(bcolors.OKGREEN + frmt.format(fifty2Test) + bcolors.ENDC)
            #print(to_print.to_string())
            #print(to_print)
        else:
            #printHOLD(' hold based on 0.95 of 52 week high')
            to_print.loc[tis,cols[3]] = frmts.format(bcolors.FAIL + frmt.format(fifty2Test) + bcolors.ENDC)
        if close<=fifty2Test2:
            #printBUY(' buy based on 0.90 of 52 week high')
            to_print.loc[tis,cols[4]] = frmts.format(bcolors.OKGREEN + frmt.format(fifty2Test2) + bcolors.ENDC)
        else:
            #printHOLD(' hold based on 0.90 of 52 week high')
            to_print.loc[tis,cols[4]] = frmts.format(bcolors.FAIL + frmt.format(fifty2Test2) + bcolors.ENDC)
        if close<=fifty2Test3:
            #printBUY(' buy based on 0.85 of 52 week high')
            to_print.loc[tis,cols[5]] = frmts.format(bcolors.OKGREEN + frmt.format(fifty2Test3) + bcolors.ENDC)
        else:
            #printHOLD(' hold based on 0.85 of 52 week high')
            to_print.loc[tis,cols[5]] = frmts.format(bcolors.FAIL + frmt.format(fifty2Test3) + bcolors.ENDC)
        # test if below moving average
        if close<=fiftyDayAverage:
            #printBUY(' buy based on 50 day average')
            to_print.loc[tis,cols[6]] = frmts.format(bcolors.OKGREEN + frmt.format(fiftyDayAverage) + bcolors.ENDC)
        else: 
            #printHOLD(' hold based on 50 day average')
            to_print.loc[tis,cols[6]] = frmts.format(bcolors.FAIL + frmt.format(fiftyDayAverage) + bcolors.ENDC)
        if close<=twoHundredDayAverage:
            #printBUY(' buy based on 200 day average')
            to_print.loc[tis,cols[7]] = frmts.format(bcolors.OKGREEN + frmt.format(twoHundredDayAverage) + bcolors.ENDC)
        else: 
            #printHOLD(' hold based on 200 day average')
            to_print.loc[tis,cols[7]] = frmts.format(bcolors.FAIL + frmt.format(twoHundredDayAverage) + bcolors.ENDC)
        # test if below previous period
        if close <= firstHistory:
            #printBUY(' buy based on below previous period')
            to_print.loc[tis,cols[8]] = frmts.format(bcolors.OKGREEN + frmt.format(firstHistory) + bcolors.ENDC)
        else:
            #printHOLD(' hold based on above previous period')
            to_print.loc[tis,cols[8]] = frmts.format(bcolors.FAIL + frmt.format(firstHistory) + bcolors.ENDC)
    #except:
        #print(' could not open ',ti,' using yfinance')
    #print(to_print)
        #printHEADER('----------------------------------------------')
ax.legend(loc='best',numpoints=1)
ax.xaxis.set_tick_params(rotation=30)
fig.show()
ax2.legend(loc='best',numpoints=1)
ax2.xaxis.set_tick_params(rotation=30)
fig2.show()
printHEADER('----------------------------------------------')
print(to_print)

input('ENTER to exit')
