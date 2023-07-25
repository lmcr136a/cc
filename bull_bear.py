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
    except Exception as error:
        print(error)
        time.sleep(3)
        coininfo = binance.fetch_ohlcv(symbol=sym, 
            timeframe=tf, since=since, limit=limit)

    df = pd.DataFrame(coininfo, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
    df.set_index('datetime', inplace=True)
    return df


def bull_or_bear(binance, sym, mode=1):  
    """
    상승장인지 하락장인지?
    사실 이게 제일 중요한거같음
    mode 1: 4시간 > 5m * (12*4=48)개
    mode 2: 12시간 > 15m * (4*12=48)개
    mode 3: 36시간 > 1h * 36개
    """
    if mode == 3:
        tf, n, a = '5m', 48, 2
    elif mode == 2:
        tf, n, a = '15m', 48, 1.5
    elif mode == 1:
        tf, n, a = '1h', 36, 1
    df = past_data(binance, sym=sym, tf=tf, limit=n)
    m = (df['high']+df['low'])/2  # df['close']로 해도 되는데 그냥 이렇게 함
    rising = []
    for i in range(1, len(m)-1):
        rising.append((m[i+1] - m[i])/m[i]*100)
    rising_coef = np.mean(rising)*a
    # print(m.keys()[0], rising_coef, len(m))
    if rising_coef > 0.04:
        return "BULL"
    elif rising_coef < -0.04:
        return "BEAR"
    else:
        return "~-~-~"
    
"""
뭘 사야할지 모르겠을땐 걍 도박으로 3% 먹고 나오기 and 6% 잃으면 걍 팔기 이런거 하자
                print("숏만가능, 기준조정")
                print("롱만가능, 기준조정")
                print("모르겠는데 하락할거같음, 만족조금만")
                print("모르겠는데 상승할거같음, 만족조금만")
                print("지그재그일거같음, 만족조금만")
                print("딴거사기")
"""
# Return short_only, long_only, buying_cond, satisfying_pnl or False
def a_l1():
    # print("롱만가능, 기준조정")
    return False, True, COND_LV2, False
def a_l2():
    # print("모르겠는데 상승할거같음, 만족조금만")
    return False, True, COND_LV1, SATISFYING_LV1
def a_s1():
    # print("숏만가능, 기준조정")
    return True, False, COND_LV2, False
def a_s2():
    # print("모르겠는데 하락할거같음, 만족조금만")
    return True, False, COND_LV1, SATISFYING_LV1
def a_x1():
    # print("지그재그일거같음, 만족조금만")
    return False, False, False, SATISFYING_LV2
def a_x2():
    # print("딴거사기")
    return False, False, 100, False



def inspect_market(binance, sym, satisfying_pnl, buying_cond=0.4, print_=True):
    st1 = bull_or_bear(binance, sym=sym, mode=1)
    st2 = bull_or_bear(binance, sym=sym, mode=2)
    st3 = bull_or_bear(binance, sym=sym, mode=3)
    if print_:
        print(f"**{sym}__[36h~ {st1}]_[12h~ {st2}]_[4h~ {st3}]")

    if st1 == 'BEAR':           # 36시간동안 하락
        if st2 == "BEAR":           # 12시간동안 하락
            if st3 == "BEAR":           # 4시간동안 하락
                actions = a_s1()
            elif st3 =="BULL":          # 하락하다가 4시간동안 갑자기 상승
                actions = a_l2()
            else:                       # 한창 떨어지고 소강상태 > 도박
                actions = a_x1()
        elif st2 == "BULL":          # 12시간동안 상승
            if st3 == "BEAR":           # 36~12 ㅈㄴ떨어짐 12부터 조금씩 오르다가 4전부터 다시 하락
                actions = a_x1()
            elif st3 == "BULL":         
                actions = a_l1()
            else:                       # 36~12하락 12부터 상승후 지그재그
                actions = a_s2()
        else:
            if st3 == "BEAR":       # 36 하락 12 지그재그 4 하락
                actions = a_s2()
            elif st3 == "BULL":     # 36 하락 12 지그재그 4 상승
                actions = a_l2()
            else:                   # 36 하락 12 지그재그 4 지그재그
                actions = a_s2()

    elif st1 == 'BULL':           # 36시간동안 상승
        if st2 == "BEAR":           # 12시간동안 하락
            if st3 == "BEAR":           # 4시간동안 하락
                actions = a_s1()
            elif st3 =="BULL":         
                actions = a_x1()
            else:                      
                actions = a_s2()
        elif st2 == "BULL":          # 36상승 12 상승
            if st3 == "BEAR":           # 도박..?
                actions = a_s2()
            elif st3 == "BULL":         # 4시간동안 상승
                actions = a_l1()
            else:                       # 36 상승 12 상승 지그재그
                actions = a_x1()
        else:
            if st3 == "BEAR":       # 36 상승 12 지그재그 4 하락
                actions = a_l2()
            elif st3 == "BULL":     # 36 상승 12 지그재그 4 상승
                actions = a_s2()
            else:                   # 36 상승 12 지그재그 4 지그재그
                actions = a_x1()

    else:                      # 36시간동안 지그재그
        if st2 == "BEAR":           # 12시간동안 하락
            if st3 == "BEAR":           # 이건 36~12 사이엔 상승하다가 12때부터 떨어지는거
                actions = a_s1()
            elif st3 =="BULL":          # 36~12 상승 12~ㅈㄴ떨어짐 4시간전부터 상승
                actions = a_l2()
            else:                       # 36~12 상승 12~ 하락 지그재그
                actions = a_x1()
        elif st2 == "BULL":          # 36~12 하락하다가 12시간동안 상승
            if st3 == "BEAR":           # 36~12 하락하다가 12시간전부터 ㅈㄴ오르다가 지금 떨어짐
                actions = a_s2()
            elif st3 == "BULL":         # 36~12 하락하다가 12시간전부터 오르기
                actions = a_l1()
            else:                       # 36~12 12~4 4~
                actions = a_s2()
        else:
            if st3 == "BEAR":       # 36 지그재그 12 지그재그 4 하락
                actions = a_s2()
            elif st3 == "BULL":     # 36 지그재그 12 지그재그 4 상승
                actions = a_l2()
            else:                   # 36 지그재그 12 지그재그 4 지그재그
                actions = a_x1()

    short_only, long_only, new_cond, new_sat_pnl = actions
    if new_cond:
        buying_cond = new_cond
    if new_sat_pnl:
        satisfying_pnl = new_sat_pnl
    
    return short_only, long_only, buying_cond, satisfying_pnl
        

    
