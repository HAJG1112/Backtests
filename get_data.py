# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 14:37:40 2020

@author: Haisun
"""
import pandas as pd
import requests
import pandas as pd

API_KEY = 'pbhRwamQlZ2MpHyxKHToX9l1fj2Igrh4wsHgZjtc'
BASE_URL = 'https://0iwb60mu47.execute-api.eu-west-2.amazonaws.com/IBContracts-Internal-Dev'


def getAllIBContracts():
    ib_contract_url = BASE_URL + '/ib_contract'
    headers = { 'x-api-key': API_KEY }
    
    response = requests.get(ib_contract_url, headers=headers)
    contracts = response.json()
    
    return contracts

def getContractDetails(contract_id):
    ib_contract_url = BASE_URL + '/ib_contract/' +str(contract_id)+ '/data?fromDate=20000101&toDate=20200505'
    headers = { 'x-api-key': API_KEY }
    response = requests.get(ib_contract_url, headers=headers)
    contract = response.json()
    
    return contract
print('='*50)

col_names = ['fxi', 'spy', 'qqq', 'eem', 'ewl', 'ewi', 'ewu', 'ewq', 'ewp', 'ewg', 'ewa', 'inda', 'iwm', 'ewz',
             'urth']

def data_type(val):
    dfs = []
    for i in range(124,139):
            x = pd.DataFrame(getContractDetails(i)).transpose()
            x.index = pd.to_datetime(x.index)
            x = pd.DataFrame(x.iloc[:,val])
            dfs.append(x)

    mom = pd.concat(dfs,axis=1)
    mom.columns = col_names
    return mom

close = data_type(0)
close.to_csv(r'C:\Users\justi\OneDrive\Haisun Documents\Jobs\Vector\Pyth\backtestv2\data\close1.csv')

high = data_type(2)
high.to_csv(r'C:\Users\justi\OneDrive\Haisun Documents\Jobs\Vector\Pyth\backtestv2\data\high1.csv')

low = data_type(3)
low.to_csv(r'C:\Users\justi\OneDrive\Haisun Documents\Jobs\Vector\Pyth\backtestv2\data\low1.csv')

open = data_type(4)
open.to_csv(r'C:\Users\justi\OneDrive\Haisun Documents\Jobs\Vector\Pyth\backtestv2\data\open1.csv')

volume = data_type(5)
volume.to_csv(r'C:\Users\justi\OneDrive\Haisun Documents\Jobs\Vector\Pyth\backtestv2\data\volume1.csv')