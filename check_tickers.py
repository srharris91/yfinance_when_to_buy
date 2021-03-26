import json
import yfinance as yf
from datetime import date,timedelta
import pandas as pd

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
today = date.today()
#d = today.strftime("%d/%m/%Y")
dend = today.isoformat()
dstart = (today - period).isoformat()

print('dstart = ',dstart)
print('dend   = ',dend)

with open('tickers.json') as f:
    data = json.load(f)
tickers = data['tickers']

printHEADER('----------------------------------------------')
for ti in tickers:
    try:
        tick = yf.Ticker(ti)
        history = tick.history(start=dstart,end=dend)
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
        printHEADER(ti)
        printHEADER(tick.info['longName'])
        to_print = pd.DataFrame(
                {
                    dend:close,
                    '52 week high':fifty2High,
                    '52 week low':fifty2Low,
                    '52 week test 0.95':fifty2Test,
                    '52 week test 0.90':fifty2Test2,
                    '52 week test 0.85':fifty2Test3,
                    '50 day average':fiftyDayAverage,
                    '200 day average':twoHundredDayAverage,
                    dstart:firstHistory 
                    },
                index=[ti,])
        print(to_print)
        if close<=fifty2Test:
            printBUY(' buy based on 0.95 of 52 week high')
        else:
            printHOLD(' hold based on 0.95 of 52 week high')
        if close<=fifty2Test2:
            printBUY(' buy based on 0.90 of 52 week high')
        else:
            printHOLD(' hold based on 0.90 of 52 week high')
        if close<=fifty2Test3:
            printBUY(' buy based on 0.85 of 52 week high')
        else:
            printHOLD(' hold based on 0.85 of 52 week high')
        # test if below moving average
        if close<=fiftyDayAverage:
            printBUY(' buy based on 50 day average')
        else: 
            printHOLD(' hold based on 50 day average')
        if close<=twoHundredDayAverage:
            printBUY(' buy based on 200 day average')
        else: 
            printHOLD(' hold based on 200 day average')
        # test if below previous period
        if close <= firstHistory:
            printBUY(' buy based on below previous period')
        else:
            printHOLD(' hold based on above previous period')
    except:
        print(' could not open ',ti,' using yfinance')
    printHEADER('----------------------------------------------')


