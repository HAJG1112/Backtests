import pandas as pd
import numpy as np
import time
import scipy
from scipy import stats
import bt
import multiprocessing
import matplotlib.pyplot as plt
import datetime as dt

class backend():

    def __init__(self):
        pass

    class indicators():      #fill with different params to initialized
        def moving_average(self, data, n):
            ma = data.rolling(n, min_periods=n).mean()  # N-mean
            return ma

        def macd(self, data, macd_n, macd_m):
            short_sma = data.rolling(macd_n).mean()
            long_sma = data.rolling(macd_m).mean()
            macd = (short_sma - long_sma)
            return (macd)

        def rsi(self, data, rsi_period):
            chg = data.diff(1)
            gain = chg.mask(chg < 0, 0)
            data['gain'] = gain
            loss = chg.mask(chg > 0, 0)
            data['loss'] = loss
            avg_gain = gain.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
            avg_loss = loss.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
            data['avg_gain'] = avg_gain
            data['avg_loss'] = avg_loss
            rs = abs(avg_gain / avg_loss)
            rsi = 100 - (100 / (1 + rs))
            return rsi

        def so_cross(self, x, so_n, so_m):  # n is the high or low window
            low = x.rolling(so_n).min()
            high = x.rolling(so_n).max()
            K = 100 * ((x - low) / (high - low))
            D = self.moving_average(K, so_m)  # we use m here to represent the D rolling window
            signal = K - D
            return signal

        def crtdr(self, high, low, close):
            crtdr = (close - low)/(high-low)
            return crtdr

        def atr(self,high, low, close, open):   # extend to take differential with respect to the underlying stock price.
            a = abs(high - low)
            b = abs(high - close)
            c = abs(low - close)
            x = pd.concat([a, b, c], axis=1)
            x['tr'] = x.max(axis=1)
            x['atr'] = x['tr'].rolling(14).mean()
            x['diff'] = x['atr'].diff()
            x['acc'] = x['diff'].diff()
            x['underlying_diff_close'] = x['diff'] / close.diff()
            x['underlying_diff_open'] = x['diff'] / open.diff()
            x['underlying_diff_gap'] = x['diff']/open-close
            return x

        def ma_rsi_3(self, data):
            rsi_3 = self.rsi(data, 3).iloc[:,0]
            ma_rsi_3 = rsi_3.rolling(3).mean()
            return pd.DataFrame(ma_rsi_3)

        def rsi_mom(self, data):
            rsi_14 = self.rsi(data, 14)
            mom = rsi_14 - rsi_14.shift(9)
            return mom

        def composite_rsi(self, data):
            rsi_mom = self.rsi_mom(data)
            rsi_3 = self.ma_rsi_3(data)
            composite = rsi_mom + rsi_3
            composite= composite.dropna()
            #df = pd.concat([rsi_mom, rsi_3], axis=0).rename(['mom', 'rsi_3'])
            return composite

    class get_values():
        def n_sample(self, data, n, i): #declare the data type given and some x-day spacing.
                slice = data.iloc[0:i]    #THIS RETURNS A CORRECT SLICE ALWAYS.
                data = slice.iloc[-1::-n]  #correct n-day slice from the bottom.
                return data[::-1]    #return the slice in the correct datetime order.

        def get_macd(self,data, n, macd_n, macd_m):
            value =[]
            for j in range(len(data)+1):
                y = gv.n_sample(data,n,j)
                d1 = i.macd(y, macd_n, macd_m)
                d1 = d1.tail(1)
                value.append(d1)
            return value

        def get_rsi(self, data, n, rsi_n):
            value = []
            for j in range(len(data)+1):
                y = gv.n_sample(data, n, j)
                d1 = i.rsi(y, rsi_n)
                d1 = d1.tail(1)
                value.append(d1)
            return value

        def get_soma(self, data, n, so_n, so_m):
            value = []
            for j in range(len(data)+1):
                y = gv.n_sample(data, n, j)
                d1 = i.so_cross(y, so_n, so_m)
                d1 = d1.tail(1)
                value.append(d1)
            return value

        def get_crtdr(self, high, low, close):
            x = i.crtdr(high, low, close)
            return x

    class scaling():
        def scaling_distribution(self, data, fc_window, i):
            dist = data[i-fc_window:i]
            dist = pd.DataFrame(dist)               #take window, turn into a matrix
            return np.diag(dist).reshape(-1,1)      #take diagonal of marix to return values as list

        def minmaxscaling(self, data, fc_window, i):
            X_train = self.scaling_distribution(data, fc_window,i)
            curr_value = data[i]
            scaled_val = (curr_value - X_train.min())/(X_train.max() - X_train.min())
            return scaled_val

        def standardization(self, data, fc_window, i):
            X_train = self.scaling_distribution(data, fc_window, i)
            curr_value = data[i]
            scaled_val = (curr_value - X_train.mean()/(X_train.var()))
            return scaled_val

    class get_weights():
        def get_macd_weight(self, data, n, macd_n, macd_m, fc_window):
            weight = []
            macd = gv.get_macd(data, n, macd_n, macd_m)
            for i in range(len(data) + 1):
                if i >= fc_window:
                    value = sc.minmaxscaling(macd, fc_window, i)                  #need flexible way to change from minmax to normalized
                    weight.append(value)
            df = pd.DataFrame(weight).transpose()
            weight = pd.DataFrame(np.diag(pd.DataFrame(weight)), index=df.index)
            return weight

        def get_rsi_weight(self, data, n, rsi_n, fc_window):
            import numpy as np
            weight = []
            rsi = gv.get_rsi(data, n, rsi_n)
            for i in range(len(data)+1):
                if i>=fc_window:
                    value = rsi[i]
                    weight.append(value)
            df = pd.DataFrame(weight).transpose()
            weight = pd.DataFrame(np.diag(pd.DataFrame(weight)), index=df.index)
            return (weight/100)

        def get_soma_weight(self, data, n, so_n, so_m, fc_window):
            weight = []
            macd = gv.get_soma(data, n, so_n, so_m)
            for i in range(len(data) + 1):
                if i >= fc_window:
                    value = sc.minmaxscaling(macd, fc_window, i)
                    weight.append(value)
            df = pd.DataFrame(weight).transpose()
            weight = pd.DataFrame(np.diag(pd.DataFrame(weight)), index=df.index)
            return weight

        def get_all_weights(self, prices, w1, w2):
    
            final = pd.DataFrame()
            count = 0
            for column in prices:
                count = count + 1
                rsi = gw.get_rsi_weight(prices[column], 1, 14, 252)
                macd = gw.get_macd_weight(prices[column], 5, 30, 60, 500)
                combined = w1*rsi + w2*macd
                print(count)
                if count <= len(prices.columns):
                    final = pd.concat([final, combined], axis=1)
            final.columns = col_names
            return(final)

#### load in our data ####
col_names = ['fxi', 'spy', 'qqq', 'eem', 'ewl', 'ewi', 'ewu', 'ewq', 'ewp', 'ewg', 'ewa', 'iwm', 'ewz']
close = pd.read_csv((r'C:\Users\justi\OneDrive\Haisun Documents\Jobs\Vector\Pyth\backtestv2\data\close1.csv'), parse_dates=True,
                   index_col=0)
volume = pd.read_csv((r'C:\Users\justi\OneDrive\Haisun Documents\Jobs\Vector\Pyth\backtestv2\data\volume1.csv'), parse_dates=True,
                   index_col=0)
open = pd.read_csv((r'C:\Users\justi\OneDrive\Haisun Documents\Jobs\Vector\Pyth\backtestv2\data\open1.csv'), parse_dates=True,
                   index_col=0)
high = pd.read_csv((r'C:\Users\justi\OneDrive\Haisun Documents\Jobs\Vector\Pyth\backtestv2\data\high1.csv'), parse_dates=True,
                   index_col=0)
low =  pd.read_csv((r'C:\Users\justi\OneDrive\Haisun Documents\Jobs\Vector\Pyth\backtestv2\data\low1.csv'), parse_dates=True,
                   index_col=0)

backend = backend()
i = backend.indicators()
gv = backend.get_values()
sc = backend.scaling()
gw = backend.get_weights()
#
print('composite index')
comp = i.composite_rsi(close[['spy']])
rsi = gv.get_rsi(close[['spy']], 1, 14)
print(rsi)
comp.plot(figsize = (20,10))
plt.savefig('_test_composite_index')

'''
#x = get_all_weights(close, 0.5, 0.5)
x = get_all_weights(close, 0.5, 0.5)
y = get_all_weights(close, 0.3, 0.7)
z = get_all_weights(close, 0.25, 0.75)
x.to_csv("Weights/InverseRSI0.5+MACD0.5.csv")
y.to_csv("Weights/InverseRSI0.3+MACD0.7.csv")
z.to_csv("Weights/InverseRSI0.25+MACD0.75.csv")

1. take the first/second order derivative of atr with respect to a change in time and wrt to a change in the value of the underlying. 
2. Add threshold on raw RSI and MACD values.
3. time to start scaling risk is due to divergence dependence, between composite index and the price. 
   This divergence triggers us to scale down or stop loss.
4. where we price that stop order is atr dependent. ATR gives us an idea of intra-day volatility.
5. 2x ATR length we can set as our stop distant. (this would be enough damage to signal a trend reversal).
6. ATR is high, close-close low, go long gamma!
7. ATR could be a high, close to close low, leading to value uncertainty.
8. When certainty goes down, intra-day swings increase!
9. will incremental news impact the sentiment of market traders.
10. what is our criteria to start trading again?
11. test size of sell-off correlated to size of divergence.
summmer 18 - early 19, blow up spy and algo on to get more detailed line, get it on the everyday side, convert into excel. add indicators underneath and sec weights. 
major divergence, delta signal!

12. Dispersion trading....RV trade, buy rv correlation on constituents sell on ETF. Correlation trading...
13. real money is made running position for longer! - net net index doesnt need to go anywhere.
14. carry trading strategy, wing protection.
15. find way to develop macro backdrop - GDP trend, yield curve, inflation/real, some macro indicators to define regime.


for time testing
        #start = time.time()
         # elapsed = (time.time() - start)
       # print("iter number: {0} , time to iter: {1}" , count, elapsed)
'''