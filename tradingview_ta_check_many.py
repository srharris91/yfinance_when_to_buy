#import tradingview_ta
from tradingview_ta import TA_Handler#, Interval, Exchange
import json
#
#print(tradingview_ta.__version__)

class bcolors:
    NEUTRAL = '\033[95m'
    #OKBLUE = '\033[94m'
    #OKCYAN = '\033[96m'
    BUY = '\033[92m'
    #WARNING = '\033[93m'
    SELL = '\033[91m'
    ENDC = '\033[0m'
    #BOLD = '\033[1m'
    #UNDERLINE = '\033[4m'


tickers_exchange_json = {}
with open('ticker_exchange.json') as ifh:
    tickers_exchange_json = json.load(ifh)
ticker_exchange = tickers_exchange_json['ticker_exchange']
intervals = tickers_exchange_json['intervals']
ratio = tickers_exchange_json['ratio']
#intervals = [
        #'1m',#Interval.INTERVAL_1_MINUTE,
        #'5m',#Interval.INTERVAL_5_MINUTES,
        #'15m',#Interval.INTERVAL_15_MINUTES,
        #'1h',#Interval.INTERVAL_1_HOUR,
        #'4h',#Interval.INTERVAL_4_HOURS,
        #'1d',#Interval.INTERVAL_1_DAY,
        ##'1W',#Interval.INTERVAL_1_WEEK,
        #'1M'#Interval.INTERVAL_1_MONTH,
        #]
keys = ['BUY','SELL','NEUTRAL']
handler = TA_Handler(
        symbol = 'VUG',
        screener = 'america',
        exchange = 'AMEX',
        interval = intervals[0]
        )
print(bcolors.NEUTRAL,'  {:10s}  {:15s}  '.format('Ticker','Recommendation'),'Summary of Technical Analysis',bcolors.ENDC)
print('--------------------------------------------------------------------------')
tickers_to_buy = []
tickers_to_sell = []
for symbol,exchange in ticker_exchange.items():
    handler.set_symbol_as(symbol)
    handler.exchange = exchange
    summary = {'BUY':0, 'SELL':0, 'NEUTRAL':0}
    for intervali in intervals:
        handler.set_interval_as(intervali)
        analysis = handler.get_analysis()
        #print('  ',analysis.interval,analysis.summary)
        #print(analysis.interval)
        #print(intervali,analysis.oscillators)
        #print(intervali,analysis.moving_averages)
        #keys = [key for key in analysis.indicators.keys() if 'Pivot' in key]
        #for key in analysis.indicators.keys():
            #if 'Pivot' in key:
                #print(intervali,key,analysis.indicators[key])
        #print('  ',summary+analysis.summary)
        for key,val in summary.items():
            summary[key] += analysis.summary[key]
        #print(analysis.time)
        #print(analysis.oscillators)
        #print(analysis.indicators)
    #print('  ',summary)
    num_tests = 0
    for summary_key,summary_val in summary.items():
        num_tests += summary_val
    shaun_recommendation = 'NEUTRAL'
    for summary_key,summary_val in summary.items():
        if summary_val>ratio*num_tests:
            shaun_recommendation = summary_key
    if shaun_recommendation == 'BUY':
        #print(bcolors.BUY+handler.symbol,shaun_recommendation,summary,bcolors.ENDC)
        print(bcolors.BUY,'  {:10s}  {:15s}  '.format(handler.symbol,shaun_recommendation),summary,bcolors.ENDC)
        tickers_to_buy.append(handler.symbol)
    elif shaun_recommendation == 'SELL':
        #print(bcolors.SELL,handler.symbol,shaun_recommendation,summary,bcolors.ENDC)
        print(bcolors.SELL,'  {:10s}  {:15s}  '.format(handler.symbol,shaun_recommendation),summary,bcolors.ENDC)
        tickers_to_sell.append(handler.symbol)
    elif shaun_recommendation == 'NEUTRAL':
        #print(bcolors.NEUTRAL,handler.symbol,shaun_recommendation,summary,bcolors.ENDC)
        print(bcolors.NEUTRAL,'  {:10s}  {:15s}  '.format(handler.symbol,shaun_recommendation),summary,bcolors.ENDC)
#print(help(handler))
#print(help(handler.get_analysis()))
#print(handler.get_indicators())

print(bcolors.BUY+'recommended to buy ',tickers_to_buy,bcolors.ENDC)
print(bcolors.SELL+'recommended to sell ',tickers_to_sell,bcolors.ENDC)
