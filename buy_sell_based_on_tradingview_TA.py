from tradingview_ta import TA_Handler#, Interval, Exchange
import sys
sys.path.append('/home/shaun/GITHUBs/investopedia_simulator_api')
from investopedia_api import InvestopediaApi, TradeExceedsMaxSharesException
import json
import pprint

# user input of sending an email about the trade?
send_email = False

# investopedia
credentials = {}
with open('credentials.json') as ifh:
    credentials = json.load(ifh)
client = InvestopediaApi(credentials)

p = client.portfolio
print("account value: %s" % p.account_value)
print("cash: %s" % p.cash)
print("buying power: %s" % p.buying_power)
buying_power = float(p.buying_power)
print("annual return pct: %s" % p.annual_return_pct)
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
        #'1W',#Interval.INTERVAL_1_WEEK,
        #'1M'#Interval.INTERVAL_1_MONTH,
        #]
def get_buy_or_sell(ticker_exchange,intervals,ratio):
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
    return tickers_to_buy,tickers_to_sell
tickers_to_buy,tickers_to_sell = get_buy_or_sell(ticker_exchange,intervals,ratio)


for pos in tickers_to_buy:
    trade1 = client.StockTrade(symbol=pos, quantity=1, trade_type='buy',
                               order_type='market', duration='good_till_cancelled', send_email=send_email)
    # validate the trade
    trade_info = trade1.validate()
    #pprint.pprint(trade_info)
    #trade_info = trade1.validate()
    price = float(trade_info['Est_Total'][1:].replace(',',''))
    if buying_power > price:
        if trade1.validated:
            pprint.pprint(trade_info)
            trade1.execute()
            buying_power -= price
# now check investopedia positions
long_positions = client.portfolio.stock_portfolio
print("--------------------")
for pos in long_positions:
    print(pos.symbol, pos.change)
    #print(pos.purchase_price)
    #print(pos.current_price)
    #print(pos.change)
    #print(pos.total_value)

    # This gets a quote with addtional info like volume
    #quote = pos.quote
    #if quote is not None:
        #print(quote.__dict__)
print("---------------------")
ticker_exchange_positions = {}
for pos in long_positions:
    ticker_exchange_positions[pos.symbol] = ticker_exchange[pos.symbol]

buy_position,sell_position = get_buy_or_sell(ticker_exchange_positions,intervals,ratio)
for pos in buy_position:
    trade1 = client.StockTrade(symbol=pos, quantity=1, trade_type='buy',
                               order_type='market', duration='good_till_cancelled', send_email=send_email)
    # validate the trade
    trade_info = trade1.validate()
    #pprint.pprint(trade_info)
    #trade_info = trade1.validate()
    price = float(trade_info['Est_Total'][1:].replace(',',''))
    if buying_power > price:
        if trade1.validated:
            pprint.pprint(trade_info)
            trade1.execute()
            buying_power -= price
for pos in sell_position:
    trade1 = client.StockTrade(symbol=pos, quantity=1, trade_type='sell',
                               order_type='market', duration='good_till_cancelled', send_email=send_email)
    # validate the trade
    trade_info = trade1.validate()
    #pprint.pprint(trade_info)
    #trade_info = trade1.validate()
    if trade1.validated:
        pprint.pprint(trade_info)
        trade1.execute()


client.refresh_portfolio()
open_orders = client.open_orders
pprint.pprint(open_orders)
