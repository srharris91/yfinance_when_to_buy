#import tradingview_ta
from tradingview_ta import TA_Handler#, Interval, Exchange
#
#print(tradingview_ta.__version__)

intervals = [
        '1m',#Interval.INTERVAL_1_MINUTE,
        '5m',#Interval.INTERVAL_5_MINUTES,
        '15m',#Interval.INTERVAL_15_MINUTES,
        '1h',#Interval.INTERVAL_1_HOUR,
        '4h',#Interval.INTERVAL_4_HOURS,
        '1d',#Interval.INTERVAL_1_DAY,
        '1W',#Interval.INTERVAL_1_WEEK,
        '1M'#Interval.INTERVAL_1_MONTH,
        ]
summary = {'BUY':0, 'SELL':0, 'NEUTRAL':0}
keys = ['BUY','SELL','NEUTRAL']
handler = TA_Handler(
        symbol = 'VUG',
        screener = 'america',
        exchange = 'AMEX',
        interval = intervals[0]
        )
#help(TA_Handler)
print(handler.symbol)
for intervali in intervals:
    handler.set_interval_as(intervali)
    analysis = handler.get_analysis()
    print('  ',analysis.interval,analysis.summary)
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
print('  ','sum',summary)
#print(help(handler))
#print(help(handler.get_analysis()))
#print(handler.get_indicators())
