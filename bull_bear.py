import datetime
import numpy as np
import pandas as pd
import mplfinance as mpf
import time
import matplotlib.pyplot as plt
import ccxt 

from HYPERPARAMETERS import *


def past_data(binance, sym, tf, limit, since=None):
    try:
        coininfo = binance.fetch_ohlcv(symbol=sym, 
            timeframe=tf, since=since, limit=limit)
    except:
        time.sleep(3)
        coininfo = binance.fetch_ohlcv(symbol=sym, 
            timeframe=tf, since=since, limit=limit)

    df = pd.DataFrame(coininfo, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
    df.set_index('datetime', inplace=True)
    return df


def bull_or_bear(binance, sym, mode=1, ref=0.04):  
    """
    상승장인지 하락장인지?
    사실 이게 제일 중요한거같음
    mode 1: 30m
    mode 2: 2h
    mode 3: 4h
    """
    mode = 3
    if mode == 3:
        tf, n, a = '1m', 30, 2
    elif mode == 2:
        tf, n, a = '5m', 24, 1.5
    elif mode == 1:
        tf, n, a = '15m', 16, 1
    df = past_data(binance, sym=sym, tf=tf, limit=n)
    m = (df['high']+df['low'])/2  # df['close']로 해도 되는데 그냥 이렇게 함
    rising = []
    for i in range(1, len(m)-1):
        rising.append((m[i+1] - m[i])/m[i]*100)

    # 그 ㅈㄴ예쁘게 올라가는게 0.142, 0.00567, 0.155
    rising_coef = np.mean(rising)*a
    if rising_coef > ref:
        return "BULL"
    elif rising_coef < -ref:
        return "BEAR"
    else:
        return "~-~-~"


def inspect_market(binance, sym, print_=True):
    st1 = bull_or_bear(binance, sym=sym, mode=1)
    st2 = bull_or_bear(binance, sym=sym, mode=2)
    st3 = bull_or_bear(binance, sym=sym, mode=3)
    if print_:
        print(f"**{sym}__[4h~ {st1}]_[2h~ {st2}]_[30min~ {st3}]")

    if st1 == 'BEAR' and st2 == "BEAR" and st3 == "BEAR":
        return SHORT
    elif st1 == 'BULL' and st2 == "BULL" and st3 == "BULL":         # 4시간동안 상승
        return LONG
                
    else:                      # 36시간동안 지그재그
        if st2 == "BEAR" and st3 == "BEAR":           # 이건 36~12 사이엔 상승하다가 12때부터 떨어지는거
            return SHORT
        elif st2 == "BULL" and st3 == "BULL":         # 36~12 하락하다가 12시간전부터 오르기
            return LONG
        
        # else:
        #     if st3 == "BEAR":       # 36 지그재그 12 지그재그 4 하락
        #         return SHORT
        #     if st3 == "BULL":     # 36 지그재그 12 지그재그 4 상승
        #         return LONG
