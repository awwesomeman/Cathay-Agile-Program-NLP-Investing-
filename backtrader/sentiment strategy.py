import backtrader as bt
import backtrader.feeds as btfeeds
import pandas as pd
import numpy as np
import datetime

# load data  ==================================================
def clean_data(stock_id=('2330','台積電')):
    semi_stock  = pd.read_csv('./data/semiconductor_daily_adj.csv',header = [0,1])
    semi_stock.index = pd.to_datetime(semi_stock.date.values.flatten())
    semi_stock.drop(['date'],axis=1,level=0,inplace=True)

    stock = semi_stock[stock_id[0]].copy()
    stock.volume = stock.volume.astype('int')
    stock.index.name='datetime'
    stock = stock[['open','high','low','close','volume']]

    # load in sentiment ==================================================
    vader = np.load('./data/gooodinfo2330_sent_log_odds.npy')
    vader = pd.Series(vader)
    news = pd.read_csv('./data/goodinfo_2330.csv')

    sentiment_df = pd.concat([news[['title','datetime']],vader],axis=1,join='inner')
    sentiment_df.set_index('datetime',inplace=True)
    sentiment_df.index = pd.to_datetime(sentiment_df.index)
    sentiment_df.columns = ['title','sentiment']
    
    sentiment_df.index = sentiment_df.index.date
    vader_daily = sentiment_df.groupby(sentiment_df.index)[['sentiment']].mean().copy()
    vader_daily = vader_daily[['sentiment']]
    # vader_daily = ((vader_daily - vader_daily.min()) / (vader_daily.max() - vader_daily.min())).to_frame('sentiment')
    vader_daily.index = pd.to_datetime(vader_daily.index)
    vader_daily.index.name = 'datetime'
    vader_daily = vader_daily.rolling(10).mean().dropna()

    return stock, vader_daily

# observer =========================================================
class MyBuySell(bt.observers.BuySell):
    plotlines = dict(
        buy=dict(marker='$\u21E7$', markersize=12.0, color='black'),
        sell=dict(marker='$\u21E9$', markersize=12.0, color='lime')
    )
    
# indicator =========================================================
class SentimentIndicator(bt.Indicator):
    lines = ('st','st_ma1','st_ma2','st_ma3','st_spread','st_spread_ma1','st_spread_ma2')
    params = (('st_ma1_period', 10),('st_ma2_period', 20),('st_ma3_period', 30),('st_spread_ma1_period', 10),('st_spread_ma2_period', 20))
    plotinfo = dict(
        # Add extra margins above and below the 1s and -1s
        # plotymargin=0.15,

        # Plot a reference horizontal line at 1.0 and -1.0
        # plothlines=[1.0, -1.0],
    )

    plotlines = dict(
                st = dict(ls='-',color='black',alpha=0.3, _name='origin'),
                st_ma1 = dict(ls='-',color='turquoise'),
                st_ma2 = dict(ls='-',color='blue'),
                st_ma3 = dict(ls='--',color='red'),
                st_spread = dict(_method='bar',alpha=0.6,color='orange'),
                )

    def __init__(self):
        # self.addminperiod(30)
        # self.plotinfo.plotmaster = self.datas[0]
        self.lines.st = bt.indicators.SMA(self.datas[0].sentiment, period=1)
        self.lines.st_ma1 = bt.indicators.SMA(self.datas[0].sentiment, period=self.params.st_ma1_period)
        self.lines.st_ma2 = bt.indicators.SMA(self.datas[0].sentiment, period=self.params.st_ma2_period)
        self.lines.st_ma3 = bt.indicators.SMA(self.datas[0].sentiment, period=self.params.st_ma3_period)

        self.lines.st_spread_ma1 = bt.indicators.SMA((self.lines.st - self.lines.st_ma3), period=self.params.st_spread_ma1_period)
        self.lines.st_spread_ma2 = bt.indicators.SMA((self.lines.st - self.lines.st_ma3), period=self.params.st_spread_ma2_period)

    def next(self):
        self.lines.st_spread[0] = self.lines.st[0] - self.lines.st_ma3[0]
        

# datafeed =========================================================
class DataFeed(bt.feeds.PandasData):
    lines = ('sentiment', ) # datetime, open, high, low, close, openinterest 皆有預設 lines，所以不用再設定參數
    params = (
        # ('datetime', None), 
        # ('open', 'open'), 
        # ('high', 'high'), 
        # ('low', 'low'), 
        # ('close', 'close'), 
        # ('openinterest', 'openinterest'), 
        ('sentiment', 'sentiment'),
        # ('fromdate', datetime.date(2018, 1, 2)),
        # ('todate', None),
    )
# strategy =========================================================
class Strategy(bt.Strategy):
    params = dict(
        verbose=False,
        # sentiment params
        st_ma1_period = 10,
        st_ma2_period = 20,
        st_ma3_period = 40,
        st_spread_ma1_period = 10,
        st_spread_ma2_period = 30,
        lookback = 20,
        # stock params
        ma1 = 20,
        ma2 = 40,
    )

    def __init__(self):
        # Sentiment scores from VADER
        self.sentiment = self.datas[0].sentiment

        # Abbriviation for simplicity
        self.dataopen = self.datas[0].open
        self.dataclose = self.datas[0].close

        # Sentiment indicator
        self.stind = SentimentIndicator(self.datas[0],
                                        st_ma1_period = self.params.st_ma1_period,
                                        st_ma2_period = self.params.st_ma2_period,
                                        st_ma3_period = self.params.st_ma3_period,
                                        st_spread_ma1_period = self.params.st_spread_ma1_period,
                                        st_spread_ma2_period = self.params.st_spread_ma2_period,
                                        subplot=True)
        # Stock indicator
        self.ma1 = bt.indicators.SMA(self.datas[0].close, period=self.params.ma1)
        self.ma2 = bt.indicators.SMA(self.datas[0].close, period=self.params.ma2)
        # self.rsi = bt.indicators.RSI_SMA(self.datas[0].close, period=21)       
         
        # Trading signals
        self.buy_sig = bt.indicators.CrossOver(self.stind.st_spread_ma1,self.stind.st_spread_ma2)
        self.buy_sig.plotinfo.plot=False

    def log(self, name, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        if self.params.verbose:
            print('%s %s %s' % (dt.isoformat(), name, txt))

    def next(self):
        # Trading filters
        # Sentiment filters
        # some value must be smaller than zero
        self.buy_filter_sent = (self.stind.st_spread_ma1[0] < 0) and (self.stind.st_spread_ma2[0] < 0)
        self.buy_filter_sent2 = sum(np.where(np.array(self.stind.st_spread_ma2.get(-1,size=self.params.lookback))<0,1,0)) >= int(self.params.lookback /2) \
                           # and sum(np.where(np.array(self.stind.st_spread_ma1.get(-1,size=self.params.lookback))<0,1,0)) >= int(self.params.lookback /2)

        # there are at leat n/2 periods that current value is smaller than previous value
        past_diff = self.stind.st_spread_ma2.get(-1,size=self.params.lookback)
        self.buy_filter_sent3 = sum(np.where(np.diff(past_diff)<0,1,0)) >= int(self.params.lookback /2)

        # Stock filters
        self.buy_filter_stock = self.ma1[0] < self.ma2[0]

        # Trading logic
        if  self.buy_sig[0]==1 \
            and self.buy_filter_sent and self.buy_filter_sent2  and self.buy_filter_sent3 \
            and self.buy_filter_stock:

            self.order = self.buy()
            print('buy')
        # ----------------------------------
        
        # print('=================')  
        # self.log()
        # print('-----------------')
        # self.log())
        # print('-----------------')
        # self.log()
        # print('-----------------')
        # self.log('ma1',self.stind.st_ma1[0])
        # print('=================')

# run module =========================================================
def main():
    stock_id=('2330','台積電')
    stock, vader_daily = clean_data(stock_id = stock_id)
    df = pd.merge(stock, vader_daily, right_index=True, left_index=True)
    df = df.reset_index(drop=False)
    data = DataFeed(dataname=df, datetime='datetime', open='open', high='high', low='low', close='close', sentiment='sentiment')

    cerebro = bt.Cerebro()
    cerebro.adddata(data, name=stock_id[0])
    bt.observers.BuySell = MyBuySell
    cerebro.addstrategy(Strategy, verbose=True)
    # cerebro.addsizer()
    # cerebro.broker.setcash(1e7)
    cerebro.broker.setcommission(commission=0.2/100)
    runstrat = cerebro.run(maxcpus=4)
    cerebro.plot(volume=False,style='candlestick', barup='red', bardown='green') 

if __name__ == '__main__':
    main()
