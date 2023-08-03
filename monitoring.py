import datetime
import numpy as np
import pandas as pd
import mplfinance as mpf
import time
import argparse
import matplotlib.pyplot as plt
import ccxt 
from utils import *


print(datetime.datetime.now())
binance = get_binance()

th1 = 0.9
th_num = 15

while 1:
    falling_list = []
    for sym in SYMLIST:
        m = past_data(binance,sym=sym, tf='1m', limit=10)['close']
        # print((m[-1]-m[-2] + m[-1]-m[-3])/m[-1]*100)
        if (m[-1]-m[-2] + m[-1]-m[-3])/m[-1]*100 < -th1:
            falling_list.append(m)
        if len(falling_list) > th_num:
            print(datetime.datetime.now(), " MARKET FALLING")
            with open('monitoring.txt', 'w') as f:
                f.write(str(True))
    print(f"{datetime.datetime.now()} Falling Coin Num: {len(falling_list)}")
    if len(falling_list) < th_num:
        with open('monitoring.txt', 'w') as f:
            f.write(str(False))
        

## 갑자기 떨어지는 애들이 생기면 1로 바뀜