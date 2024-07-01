# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 11:41:47 2024

@author: Jun Hui
"""

##Pulling data from yfinance
import datetime as dt
import yfinance as yf
import pandas as pd

stocks = ["AMZN", "MSFT", "GOOG", "META", "TSLA", "COIN"]
start = dt.datetime.today()-dt.timedelta(3650)
end = dt.datetime.today()
cl_price = pd.DataFrame()
ohlcv_data = {}

for ticker in stocks:
    cl_price[ticker] = yf.download(ticker,start,end)["Adj Close"]
    
##addressing NaN values (dropna, bfill, ffill)
cl_price.bfill(axis=0)

## % return of each of the columns
#Method 1 - Using pandas pct_change
daily_return = cl_price.pct_change()

#Method 2 - calculating it manually
daily_returnm2 = (cl_price/cl_price.shift(1) - 1)

##understanding the data (mean,std,median,head,tail)
#cl_price.describe()
data_describe = daily_return.describe()
### a reasonable pick would be a stock with higher stable returns (mean) and low volatility (std)

##Statistics over a rolling window
#Using rolling mean filters out the noise, gives us a smoother curve
#simple moving average of 10 days
daily_return.rolling(window=10).mean()
daily_return.rolling(window=10).std()

#exponential moving average
daily_return.ewm(com=10, min_periods=10).mean()

