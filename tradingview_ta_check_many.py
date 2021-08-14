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
#amex='AMEX'
#nyse='NYSE'
#nasdaq='NASDAQ'
#ticker_exchange = {
        #'VBK':amex,     # VSGAX ETF small cap growth
        #'VT':amex,      # VTWAX ETF total world index
        #'VUG':amex,     # large cap growth index
        #'VOO':amex,     # S&P 500 index ETF
        #'VGT':amex,     # technology index etf
        #'QQQ':nasdaq,   # invesco trust series 1
        #'TQQQ':nasdaq,  # ProShares UltraPro QQQ leveraged
        #'AAPL':nasdaq,  # ProShares UltraPro QQQ leveraged
        #'FB':nasdaq,    # Facebook
        #'NVDA':nasdaq,  # Nvidia
        #'MSFT':nasdaq,  # microsoft
        #'AMZN':nasdaq,  # amazon
        #'GOOGL':nasdaq, # google
        #'GOOG':nasdaq,  # google
        #'TSLA':nasdaq,  # tesla
        #'V':nyse,       # Visa
        #'PYPL':nasdaq,  # paypal
        #'HD':nyse,      # Home depot
        #'MA':nyse,      # Mastercard
        #'DIS':nyse,     # Disney
        #'ADBE':nasdaq,  # Adobe
        #'NFLX':nasdaq,  # netflix
        #'CRM':nyse,     # salesforce
        #'TMO':nyse,     # Thermo Fisher Scientific
        #'NKE':nyse,     # Nike
        #'ACN':nyse,     # Accenture Plc
        #'TXN':nasdaq,   # Texas Instruments Incorporated
        #'COST':nasdaq,  # Costco
        #'MCD':nyse,     # McDonald's
        #'LIN':nyse,     # Linde PLC
        #'UNP':nyse,     # Union Pacific Corporation
        #'BA':nyse,      # Boeing Co
        #'LOW':nyse,     # Lowe's
        #'SBUX':nasdaq,  # Starbucks corporation
        #'AMAT':nasdaq,  # Applied Materials
        #'INTU':nasdaq,  # Intuit Inc.
        #'AMT':nyse,     # American Tower Corp
        #'AMD':nasdaq,   # Advanced Micro Devices
        #'ISRG':nasdaq,  # Intuitive Surgical 
        #'NOW':nyse,     # ServiceNow
        #'SPGI':nyse,    # S&P Global Inc.
        #'SQ':nyse,      # Square Inc.
        #'MU':nasdaq,    # Micron Technology Inc.
        #'LRCX':nasdaq,  # Lam Research Corp.
        #'AVGO':nasdaq,  # Broadcom Inc.
        #'BKNG':nasdaq,  # Booking Holdings
        #'ZTS':nyse,
        #'CHTR':nasdaq,
        #'FIS':nyse,
        #'DHR':nyse,
        #'ADP':nasdaq,
        #'CCI':nyse,
        #'SYK':nyse,
        #'QCOM':nasdaq,
        #'MRNA':nasdaq,
        #'SNAP':nyse,
        #'ZM':nasdaq,
        #'TJX':nyse,
        #'UPS':nyse,
        #'ATVI':nasdaq,
        #'EQIX':nasdaq,
        #'EL':nyse,
        #'ILMN':nasdaq,
        #'CL':nyse,
        #'UBER':nyse,
        #'SHW':nyse,
        #'EW':nyse,
        #'ADSK':nasdaq,
        #'FISV':nasdaq,
        #'TWLO':nyse,
        #'SNOW':nyse,
        #'BSX':nyse,
        #'MCO':nyse,
        #'REGN':nasdaq,
        #'GPN':nyse,
        #'IDXX':nasdaq,
        #'ECL':nyse,
        #'ROKU':nasdaq,
        #'VRTX':nasdaq,
        #'TWTR':nyse,
        #'DG':nyse,
        #'DOCU':nasdaq,
        #'AON':nyse,
        #'CRWD':nasdaq,
        #'KLAC':nasdaq,
        #'ROP':nyse,
        #'MRVL':nasdaq,
        #'IQV':nyse,
        #'ALGN':nasdaq,
        #'A':nyse,
        #'WDAY':nasdaq,
        #'PSA':nyse,
        #'LMT':nyse,
        #'ROST':nasdaq,
        #'CMG':nyse,
        #'LULU':nasdaq,
        #'PINS':nyse,
        #}
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
    elif shaun_recommendation == 'NEUTRAL':
        #print(bcolors.NEUTRAL,handler.symbol,shaun_recommendation,summary,bcolors.ENDC)
        print(bcolors.NEUTRAL,'  {:10s}  {:15s}  '.format(handler.symbol,shaun_recommendation),summary,bcolors.ENDC)
#print(help(handler))
#print(help(handler.get_analysis()))
#print(handler.get_indicators())

print(bcolors.BUY+'recommended to buy ',tickers_to_buy,bcolors.ENDC)
