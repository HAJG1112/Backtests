import pandas as pd
import numpy as np
import scipy
from scipy import stats
import bt
import multiprocessing
import matplotlib.pyplot as plt
#### load in our data ####
col_names = ['fxi', 'spy', 'qqq', 'eem', 'ewl', 'ewi', 'ewu', 'ewq', 'ewp', 'ewg', 'ewa', 'inda', 'iwm', 'ewz', 'urth']
close = pd.read_csv((r'C:\Users\justi\OneDrive\Haisun Documents\Jobs\Vector\Pyth\backtestv2\data\close1.csv'), parse_dates=True,
                   index_col=0)
volume = pd.read_csv((r'C:\Users\justi\OneDrive\Haisun Documents\Jobs\Vector\Pyth\backtestv2\data\volume1.csv'), parse_dates=True,
                   index_col=0)
open = pd.read_csv((r'C:\Users\justi\OneDrive\Haisun Documents\Jobs\Vector\Pyth\backtestv2\data\open1.csv'),parse_dates=True,
                   index_col=0)
high = pd.read_csv((r'C:\Users\justi\OneDrive\Haisun Documents\Jobs\Vector\Pyth\backtestv2\data\high1.csv'), parse_dates=True,
                   index_col=0)
low =  pd.read_csv((r'C:\Users\justi\OneDrive\Haisun Documents\Jobs\Vector\Pyth\backtestv2\data\low1.csv'), parse_dates=True,
                   index_col=0)

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
            return x

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

        def get_rsi_three_average(self, data):
            rsi = pd.DataFrame(self.get_rsi(data, 1, 3))
            rsi = rsi.iloc[1:]
            rsi = pd.DataFrame(np.diag(pd.DataFrame(rsi)), index=rsi.columns)
            rsi = rsi.rolling(3).mean()
            rsi = rsi.dropna()
            return rsi

        def get_rsi_momentum(self, data):
            rsi = pd.DataFrame(self.get_rsi(data, 1, 14))
            rsi = rsi.iloc[1:]
            rsi = pd.DataFrame(np.diag(pd.DataFrame(rsi)), index=rsi.columns)
            rsi = rsi.dropna()
            mom = []
            for j in range(len(rsi)+1):  #loop through and take curr rsi i rsi 9 days ago, fill as current value
                curr_rsi = rsi[j]
                mom.append(curr_rsi)
            return mom

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


  #In order to make the standardization work, we need to standardzie the values, then min max scale the value according to the maximum and minimum of the standardized values, we w

        def standardization(self, data, fc_window, i):                   #apply standardization, then scale towards feature = (-1,1)
            X_train = self.scaling_distribution(data, fc_window, i)
            curr_value = data[i]
            scaled_val = (curr_value - X_train.mean()/(X_train.var()))
            return scaled_val

    class get_weights():
        def get_macd_weight(self, data, n, macd_n, macd_m, fc_window):
            standardized = pd.DataFrame()
            weight = []
            macd = gv.get_macd(data, n, macd_n, macd_m)
            for i in range(len(data) + 1):

                if i >= fc_window:
                    stand = sc.standardization(macd, fc_window,i)
                    standardized = standardized.append(stand, ignore_index=False)
                    #value = sc.minmaxscaling(standardized, fc_window, i)                  #need flexible way to change from minmax to normalized
                    #weight.append(value)
            #df = pd.DataFrame(weight).transpose()
            #weight = pd.DataFrame(np.diag(pd.DataFrame(weight)), index=df.index)
            return standardized

        def get_rsi_weight(self, data, n, rsi_n, fc_window):
            weight = []
            macd = gv.get_rsi(data, n, rsi_n)
            for i in range(len(data)+1):
                if i>=fc_window:
                    value = sc.minmaxscaling(macd, fc_window, i)
                    weight.append(value)
            df = pd.DataFrame(weight).transpose()
            weight = pd.DataFrame(np.diag(pd.DataFrame(weight)), index=df.index)
            return weight

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

    class asset_management():
        def stop_loss(self, weight, stop_level, curr_level):
            if stop_level<=curr_level:
                weight = 0
            else:
                weight = weight
            return weight, len(weight)

        def threshold(self, matrix, Llimit, Ulimit):
            matrix[matrix <= Llimit] = 0
            matrix[matrix >= Ulimit] = 1
            return matrix

        def select_top(self, matrix, N, Llimit, Ulimit):
            matrix = self.threshold(matrix, Llimit, Ulimit)
            matrix = matrix.mask(matrix.rank(axis=1, method='min', ascending=False) > N, 0)
            return matrix

backend = backend()
i = backend.indicators()
gv = backend.get_values()
sc = backend.scaling()
gw = backend.get_weights()
am = backend.asset_management()

high = high['spy']
close = close['spy']
low = low['spy']
open = open['spy']

#y = gw.get_rsi_weight(close, 1, 14, 252)
x = gw.get_macd_weight(close, 5, 30, 60, 500)
print(x)
#comb = 0.3*y + 0.7*x

#x.to_csv(r"full_macd_standardized")

'''
1. take the first/second order derivative of atr with respect to a change in time and wrt to a change in the
    value of the underlying. 
'''