import backtrader as bt
import backtrader.feeds as btfeeds
import pandas as pd
import numpy as np
import datetime

def combine_sentiment(stock_name):

  news_ = news[news.Tag.apply(lambda x: stock_name in x)]
  sentiment_df = pd.concat([news_[['Title','Content','Date']],vader],axis=1,join='inner')
  sentiment_df.set_index('Date',inplace=True)
  sentiment_df.index = pd.to_datetime(sentiment_df.index)

  return sentiment_df

# load in stock price ==================================================
stock_id = '2330'

semi_stock  = pd.read_csv('./data/semiconductor_daily_adj.csv',header = [0,1])
semi_stock.index = pd.to_datetime(semi_stock.date.values.flatten())
semi_stock.drop(['date'],axis=1,level=0,inplace=True)

stock = semi_stock[stock_id].copy()
stock.volume = stock.volume.astype('int')
stock.index.name='datetime'
stock = stock[['open','high','low','close','volume']]
stock['openinterest'] = 0

# load in sentiment ==================================================

vader = pd.read_csv('./data/cynes_sector_en_vadersentiment.csv')
vader.drop(['Unnamed: 0'],axis=1,inplace=True)

news = pd.read_csv('./data/cynes_sector.csv')
news.drop(['Unnamed: 0'],axis=1,inplace=True)

sentiment_df = combine_sentiment('台積電')

sentiment_df.index = sentiment_df.index.date
vader_daily = sentiment_df.groupby(sentiment_df.index)[['neg','pos']].mean().copy()
vader_daily['logodds'] = np.log((vader_daily['pos']+1)/(vader_daily['neg']+1))
vader_daily = vader_daily['logodds']
vader_daily = ((vader_daily - vader_daily.min()) / (vader_daily.max() - vader_daily.min())).to_frame('close')
vader_daily.index = pd.to_datetime(vader_daily.index)
vader_daily.index.name = 'datetime'

# strategy =========================================================
class sentiment_indicator(bt.Indicator):
    lines = ('st_ma1','st_ma2',)
    params = (('period_ma1', 5),('period_ma2', 10),)

    def __init__(self):
        self.addminperiod(self.params.period_ma2)
        self.plotinfo.plotmaster = self.data

    def next(self):
        self.lines.st_ma1 = bt.indicators.SMA(self.data, period=self.params.period_ma1)
        self.lines.st_ma1 = bt.indicators.SMA(self.data, period=self.params.period_ma2)


class MyStrategy(bt.Strategy):

    def __init__(self):
        self.st_ma1 = bt.indicators.SMA(self.data1, period=5,plotmaster=True)
        self.st_ma1 = bt.indicators.SMA(self.data1, period=10,plotmaster=True)
        self.data1.plotinfo.plotmaster = None
        self.data1.plotinfo.subplot = None

    def start(self):
        print('start')
    
    def prenext(self):
        print('not ready')

    def nex(self):
        print('new bar')
        print(self.data.close[0]) # 0 represent current moment










# run module =========================================================
cerebro = bt.Cerebro()


stock_daily = bt.feeds.PandasData(dataname = stock,
                                # fromdate = datetime.datetime(2018,1,2),
                                # todate = datetime.datetime(2018,1,2),
                                #open=0, high=1, low=2, close=3, volume=4, openinterest=5
)

vader_daily = bt.feeds.PandasData(dataname = vader_daily,
                                # fromdate = datetime.datetime(2018,1,2),
                                # todate = datetime.datetime(2018,1,2),
                                open=None, high=None, low=None, close=0, volume=None, openinterest=None
)




cerebro.adddata(stock_daily)
cerebro.adddata(vader_daily)
cerebro.addstrategy(MyStrategy)
cerebro.run()
cerebro.plot(volume=False)