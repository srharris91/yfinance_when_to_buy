import json
import yfinance as yf
from datetime import date,timedelta

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

print('----------------------------------------------')
for ti in tickers:
    tick = yf.Ticker(ti)
    history = tick.history(start=dstart,end=dend)
    fifty2High = tick.info['fiftyTwoWeekHigh']
    fifty2Low  = tick.info['fiftyTwoWeekLow']
    close  = tick.info['previousClose']
    firstHistory = history['Close'][0]
    # tests to buy or hold
    # test if 0.9 of 52 week high
    print(ti,tick.info['longName'])
    if close<=(fifty2High - 0.1*(fifty2High-fifty2Low)):
        printBUY(' buy based on 0.9 of 52 week high')
    else:
        printHOLD(' hold based on 0.9 of 52 week high')
    # test if below previous period
    if close <= firstHistory:
        printBUY(' buy based on below previous period')
    else:
        printHOLD(' hold based on above previous period')
    print('----------------------------------------------')


