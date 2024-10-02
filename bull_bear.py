import datetime
import numpy as np
import pandas as pd
import mplfinance as mpf
import time
import matplotlib.pyplot as plt
import ccxt 

from HYPERPARAMETERS import *


async def past_data(binance, sym, tf, limit, since=None):
    coininfo = await binance.fetch_ohlcv(symbol=sym, 
        timeframe=tf, since=since, limit=limit)

    df = pd.DataFrame(coininfo, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
    df.set_index('datetime', inplace=True)
    return df


async def bull_or_bear(binance, sym, mode=1, ref=0.05):  
    if mode == 2:
        tf, n, a = '1m', 20, 1.5
    elif mode == 1:
        tf, n, a = '1m', 60, 1
    df = await past_data(binance, sym=sym, tf=tf, limit=n)
    m = (df['high']+df['low'])/2  # df['close']로 해도 되는데 그냥 이렇게 함
    rising = []
    for i in range(1, len(m)-1):
        rising.append((m[i+1] - m[i])/m[i]*100)

    # 그 ㅈㄴ예쁘게 올라가는게 0.142, 0.00567, 0.155
    rising_coef = np.mean(rising)*a
    if rising_coef > ref:
        return "BULL", rising_coef
    elif rising_coef < -ref:
        return "BEAR", rising_coef
    else:
        return "~-~-~", rising_coef


async def inspect_market(binance, sym, print_=True):
    st1, score1 = await bull_or_bear(binance, sym=sym, mode=1)
    st2, score2 = await bull_or_bear(binance, sym=sym, mode=2)
    score = score1+score2
    if print_:
        print(f"**{sym}__[3h~ {st1}]_[30min~ {st2}]")

    if st1 == 'BEAR' and st2 == "BEAR":
        return SHORT, score
    elif st1 == 'BULL' and st2 == "BULL":         # 4시간동안 상승
        return LONG, score
    return None, 0
        # else:
        #     if st3 == "BEAR":       # 36 지그재그 12 지그재그 4 하락
        #         return SHORT
        #     if st3 == "BULL":     # 36 지그재그 12 지그재그 4 상승
        #         return LONG
