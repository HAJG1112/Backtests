import bt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## extend bt.algos.class
# extend logic to wrap our weight matrix calculation into the strategy logic

##  Set up the data that we wish to perform our backtests on
########
#https://stackoverflow.com/questions/19324453/add-missing-dates-to-pandas-dataframe

etfs = ['fxi', 'spy', 'qqq', 'eem', 'ewl', 'ewi', 'ewu', 'ewq', 'ewp', 'ewg', 'ewa', 'iwm', 'ewz']

matrix = pd.read_csv(r"C:\Users\justi\OneDrive\Haisun Documents\Jobs\Vector\Pyth\backtestv2\code\all_sec_0.3_0.7", parse_dates=True, index_col=0)
matrix = matrix.shift(1)  #must shift one day forward to make sure weight is used on the next day open
matrix = matrix.drop(matrix.columns[-1], axis=1)

open = pd.read_csv((r'C:\Users\justi\OneDrive\Haisun Documents\Jobs\Vector\Pyth\backtestv2\data\open1.csv'), parse_dates = True, index_col=0)
open = open.iloc[len(open) - len(matrix):]    #this trims down the prices to fit the weights matrix

open = open.loc['2013-03-26':'2020-01-24']
matrix = matrix.loc['2013-03-26':'2020-01-24']

class asset_management():
    def stop_loss(self, weight, stop_level, curr_level):
        if stop_level <= curr_level:
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

    def bt_weight(self, weights):
        weights['scale'] = weights.sum(axis = 1)
        weights = weights.div(weights['scale'], axis = 'index')
        weights= weights.iloc[:,:-1]

        return weights

class Backtest():
    def stop_loss(self):
        pass

    def backtest1(self, weights, price, name = 'asset'):  #must be fed a datetime index dataframe
        s = bt.Strategy(name , [bt.algos.WeighTarget(weights),
                                bt.algos.SelectAll(),
                                bt.algos.Rebalance()])
        t = bt.Backtest(s, price, initial_capital=100000)
        return t

    def backtest2(self, weights, price, name = 'asset'):  #must be fed a datetime index dataframe
        s = bt.Strategy(name, [bt.algos.WeighTarget(weights),
                                     bt.algos.SelectAll(),
                                     bt.algos.SelectMomentum(4, lookback=pd.DateOffset(months=3), sort_descending=True),
                                     bt.algos.Rebalance()])
        t = bt.Backtest(s, price, initial_capital=100000)
        return t

    def benchmark(self, price, name = 'benchmark'):   # must be fed a datetime indexed dataframe
        s = bt.Strategy(name, [bt.algos.RunOnce(),
                           bt.algos.SelectAll(),
                           bt.algos.WeighEqually(),
                           bt.algos.Rebalance()])
        return bt.Backtest(s, price, initial_capital=100000)

######
am = asset_management()

# only trades weekly. lets see if that work
mom_s = bt.Strategy('momentum', [bt.algos.WeighTarget(matrix[etfs]),
                              bt.algos.SelectAll(),
                              bt.algos.SelectMomentum(3),
                              bt.algos.WeighERC(),
                              bt.algos.Rebalance()],
                              etfs)
t = bt.Backtest(mom_s, open[etfs])
r = bt.run(t)
r.display()
r.plot()
plt.savefig("output/ETF_portfolio_SM_4_erc_weekly.png")
r.plot_security_weights(backtest = 'momentum', filter = None, figsize=(15,10))
plt.savefig("output/ETF_portfolio_SM_4__erc_weekly_sec_weights.png")
#### calls and runs all relevant backtests
#backtest = Backtest()
#data = matrix[['ewz']]
#weight = am.threshold(data, 0.2, 1)
#spy = backtest.backtest1(weight, open[['ewz']], name = 'ewz')
#bench = backtest.benchmark(open[['ewz']], name = 'benchmark')
#res = bt.run(spy, bench)


'''
#res.plot()
#plt.savefig("Backtest momentum 03/03/2020.png")
#res.display()
#res.plot_weights(backtest = 'momentum', filter = None, figsize=(15,10))  #monthly display
#res.plot_security_weights(backtest = 'momentum', filter = None, figsize=(15,10))
#plt.savefig("Backtest security weights 03/03/2020.png")
#res.plot_histograms(backtest = 'momentum')
#plt.savefig("Backtest histogram 03/03/2020.png")

######### ideas for short
# 1. cant short etf's when they drop too much, market seizes (there is a reason behind this)

####imran notes####
# 1. see distribution of weights for each indicator and their total
# 2. add in 9 plots for each of the indicator values that go into the dataframe.
# 3. overlap each graph with different colour over each
# 4. Hierarchical structure,from long, mid to daily.
# 5. Scatter plot of daily return, weekly return, monthly return against the 3 indicators.
# 6. Observe the asset allocation over the long
# 7. rank all 15 stocks on realised volatility (safest have lowest volatility)  weighinvvol
# 7. long only URTH as the benchmark

##### extra
# 1. transform all values into log first
# 2. combined plots of data and algo 'bt.merge(data,
# 3. CPU speed is too slow -  remedy by running each instance of daily, weekly and monthly on 3 different threads.
### https://pymotw.com/2/multiprocessing/basics.html
'''