
import numpy as np
import pandas as pd
import mplfinance as mpf
import time
import matplotlib.pyplot as plt
import ccxt 
from colorama import Fore, Style, init
from bull_bear import *
from datetime import datetime
from HYPERPARAMETERS import *



"""
ununun\
       |
        \
         |   nu
          `^'
"""
def fell_and_reviving(zzdic, curr_mvmt, last_diff):
    where_h, where_l = zzdic['where_h'], zzdic['where_l']
    if np.all(where_h[-3:-1] == 1) and where_l[-1]==1 and np.all(where_l[1:-1] == 0):
        # 앞선 2개 봉이 하락한것이어야하고 last_diff< CALM 이어야한다
        if curr_mvmt == FALLING and last_diff < CALM:
            return {"position":LONG,
                    "force_pnl": None}
        

"""
nununuu\
        |
        |
"""
def start_to_fall(m1, m3, m4, ref=0.01):
    # 0718 기준 ref=0.01이면 적당해보임
    # 딱 이 봉만 기준치 이상이어야 함
    pre3, pre2, pre1, now = m1[-4:]
    
    def diff(n, p):
        return (n-p)/p
    if not (now < pre1 < pre2 < pre3):
        return None
    if diff(pre2, pre3) > -ref*0.3 and diff(pre1, pre2) > -ref*0.5 and diff(now, pre1) < ref:
        if m3[-1] < 0 and m4[-1] < 0:
            return {"position":LONG,
                    "force_pnl": SATISFYING_LV2}


"""                          nununu\
                           /         \
        /nununu\         /            \
     /nu        \un     / 
    /               \unu
  /
/
"""

def camel_back():
    pass