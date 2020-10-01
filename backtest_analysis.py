import pandas as pd
import numpy as np
import sklearn
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

#Get the transaction data from the backtest
data = pd.read_csv(r"output/ewi_rsi_macd_0.3_0.7_transactions", index_col = 0)

def one_security_analysis(data):
    quantity = data.iloc[:,2]
    price = data.iloc[:,1]
    data['total_assets'] = quantity.cumsum()
    data['value'] = price*data['total_assets']
    data['daily_returns'] = np.log(data.price) - np.log(data.price.shift(1))

    return data

class VaR(object):
    pass


strategy = one_security_analysis(data)
value = strategy.value
plt.plot(value.index, value.values)
plt.show()
value.to_csv(r"value_test")
test = strategy.daily_returns.iloc[1:]
test.to_csv(r"test")

plt.hist(test, bins = 100)
plt.title('Histogram of log-returns')
plt.xlabel('log returns')
plt.show()
skew = stats.skew(test)
kurtosis = stats.kurtosis(test)
print(skew, kurtosis)


'''
currently to work on...
    1. Draw samples from the return using replacement
    2. Bootstrap these results to get some form of Gaussian distribution
    3. Perform VaR calculations


future work on the risk side
1. Use this to perform all necessary VaR calculations needed for the given portfolio, this will later become the 
    module known as "Risk Analysis"
2. Perform VaR calculations under all types of underlying distributions
3. Perform the capital adequacy requirements needed under Basel 3 in which to base risk of
4. Use this method to make sure that the risk being taken at each return is relfected in a change in the cash position 
    held for each given day
'''