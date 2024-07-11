from binance import Client
import numpy as np
import seaborn as sns
import pandas as pd
import sys
import os
#   from xgboost import XGBRegressor
sys.path.append(os.path.join('C:/', 'Users', 'Fedot', 'Downloads', 'LSTM-Crypto-Price-Prediction', 'historical_data'))


def calc_color_prob(method, candles):
    longtails = pd.DataFrame()
    y = []

    for i in range(candles.shape[0] - 1):
        if method(candles.iloc[i]):
            longtails.append(candles.iloc[i])
            y.append(1 if candles.iloc[i + 1]['open'] > candles.iloc[i + 1]['close'] else 0)
    s = 0
    for i in y:
        s += i
    print(method.__name__ + " result" + str(float(s) / float(len(y))))


def isredlongtail(o, c, h, lo):
    if o > c and h-o < c-lo:
        return True
    else:
        return False


def isredlonghead(o, c, h, lo):
    if o > c and (h-o) > c - lo:
        return True
    else:
        return False


def isgreenlonghead(o, c, h, lo):
    if o < c and h - c > o - lo:
        return True
    else:
        return False


def isgreenlongtail(o, c, h, lo):
    if o < c and h - c < o - lo:
        return True
    else:
        return False


def get_data_files(start, end, interval, currency):
    intervals = {240: '4h', 60: '1h', 30: '30m', 15: '15m', 5: '5m'}
    trading_pair = f'{currency}USDT'

    # load key and secret and connect to API
    keys = open('historical_data/binance_key1.txt').readline()
    print('Connecting to Client...')
    api = Client(keys[0], keys[1])

    # fetch desired candles of all data
    print('Fetching data (may take multiple API requests)')
    hist = api.get_historical_klines(trading_pair, intervals[interval], start_str=start, end_str=end)

    print('Finished.')

    # create numpy object with closing prices and volume
    hist = np.array(hist, dtype=np.float32)

    op = hist[:, 1]
    hi = hist[:, 2]
    lo = hist[:, 3]
    cl = hist[:, 4]
    vol = hist[:, 5]
    res = []
    for z in zip(op, cl):
        res.append((z[0] + z[1]) / 2)
    candles = pd.DataFrame({'open': op, 'close': cl, 'high': hi, 'low': lo, 'mean': res, 'vol': vol})
    #   calc_color_prob(isgreenlongtail)
    #   calc_color_prob(isgreenlonghead)
    #   calc_color_prob(isredlongtail)
    #   calc_color_prob(isredlonghead)
    dates = pd.date_range(start=start, periods=len(hist), freq="h")
    candles['Date'] = np.array(dates.to_pydatetime(), dtype=np.datetime64)

    candles.to_hdf('historical_data/candles.h5', key='candles', mode='w')
    print("\nDatapoints:  {0}".format(hist.shape[0]))
    print("Memory:      {0:.2f} Mb\n".format(hist.nbytes/1000000))

    return candles
