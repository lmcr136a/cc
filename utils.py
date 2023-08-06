
import numpy as np
import pandas as pd
import time
import random
import matplotlib.pyplot as plt
import ccxt 
from colorama import Fore, Style, init
from bull_bear import *
from datetime import datetime
from HYPERPARAMETERS import *

# init(convert=True)

def cal_compound_amt(wallet_usdt, lev, price, symnum):
    
    k = np.random.randint(low=12, high=20, size=1)
    k = k/20   # 0.6 ~ 0.95
    return np.float(wallet_usdt*lev/float(price)*k/float(symnum))


def curr_states_other_minions(binance, ref_num=6):
    balance = get_balance(binance)
    positions = balance['info']['positions']
    
    n1, n2 = 0, 0
    for position in positions:
        amt = float(position['positionAmt'])
        if amt > 0:
            n1 += 1
        elif amt < 0:
            n2 += 1
    short_only_strong, long_only_strong = False, False
    if n1 >= ref_num:
        short_only_strong = True
        # print("Short Only..")
    if n2 >= ref_num:
        long_only_strong = True
        # print("Long Only...")
        
    with open("files/before_sym.txt", 'r') as f:
        befores = f.read()
    befores = befores.split("\n")
    return short_only_strong, long_only_strong, befores


def long_cond(m1, cond=0.5, hour=3, tf_=3):
    t = int(round(hour*60/tf_))
    m1 = m1[-t:]
    m1 = minmax(m1)
    if m1[-1] < cond:  # 너무 높을때 long사는거 지양
        print(m1[-1], cond)
        return LONG
        
def short_cond(m1, cond=0.5, hour=3, tf_=3):
    t = int(round(hour*60/tf_))
    m1 = m1[-t:]
    m1 = minmax(m1)
    if m1[-1] > -cond:  # 너무 낮을때 short사는거 지양
        print(m1[-1], -cond)
        return SHORT
    

def shape_info(m, n=4):
    # return 오목/볼록, 증가/감소
    m = m[-n:]
    d_m = np.diff(m)
    dd_m = np.diff(d_m)
    if np.all(d_m < 0):         # 감소
        if np.all(dd_m < 0):    # 볼록
            return DECREASING_CONVEX
    elif np.all(d_m > 0):
        if np.all(dd_m > 0):
            return INCREASING_CONCAVE


def curr_movement(m, minute=2, ref=0.02):
    diff = []
    m = m[-minute-1:]
    for i in range(len(m)-1):
        diff.append(m[i+1] - m[i])
    d = np.sum(diff)*100

    diff = (m[-1] - m[0])/m[-2]*100
    last_diff = (m[-1] - m[0])/m[-2]*100
    if d > ref*minute/2 and (last_diff > ref):
        return RISING, diff
    elif d < -ref*minute/2 and (last_diff < -ref):
        return FALLING, diff
    else:
        return "~", diff


def isitsudden(m1, status, ref=0.085):
    now = m1[-1]
    prev = m1[-2]
    percent = (now - prev)/prev*100
    # print(percent, ref)
    if status == LONG and percent > ref:
        return True
    elif status == SHORT and percent < -ref:
        return True
    return False

def get_ms(binance, sym, tf, limit, wins):
    try:
        df = past_data(binance, sym=sym, tf=tf, limit=limit)
        m1 = df['close'].rolling(window=wins[0]).mean()
        m2 =  df['close'].rolling(window=wins[1]).mean()
        m3 = df['close'].rolling(window=wins[2]).mean()
        m4 = df['close'].rolling(window=wins[3]).mean()
    except Exception as e:
        print(e)
        m1, m2, m3, m4 = get_ms(binance, sym, tf, limit, wins)

    return m1, m2, m3, m4


def get_curr_pnl(binance, sym):
    wallet = get_balance(binance)

    positions = wallet['info']['positions']
    for pos in positions:
        if pos['symbol'] == sym:
            try:
                pnl = float(pos['unrealizedProfit'])/abs(float(pos['positionAmt']))/float(pos['entryPrice'])*100*float(pos['leverage'])
            except:
                print(float(pos['unrealizedProfit']), abs(float(pos['positionAmt'])), float(pos['entryPrice']), float(pos['leverage']))
                pnl = 0
            return round(pnl,2), round(float(pos['unrealizedProfit']), 2)


def log_wallet_history(balance):
    today = datetime.now()
    try:
        wallet_info = np.load('files/wallet_log.npy')
        wallet_info = np.concatenate(
                            [wallet_info, 
                            [[time.time()], 
                            [float(balance['info']['totalWalletBalance'])],
                            [float(balance['info']['totalMarginBalance'])] ]],
                            axis=1
                            )  ## Date
    except FileNotFoundError:
        wallet_info = np.array([[time.time()], [float(balance['info']['totalWalletBalance'])], [float(balance['info']['totalMarginBalance'])]])
    np.save('files/wallet_log.npy', wallet_info)
    plt.figure()
    plt.plot(wallet_info[0], wallet_info[1], 'k-')
    plt.plot(wallet_info[0], wallet_info[2], 'b-')
    plt.savefig("wallet_log.png")
    plt.close()


def minmax(m):
    m = (m - m.min())/(m.max() - m.min())
    # 원래 0~1 사이인데, 그냥 -1~1로 하고싶음
    m = 2*(m-0.5)
    return m

def inv_minmax(m, val):
    original_val = (val/2+0.5)*(m.max() - m.min())+m.min()
    return original_val


def get_balance(binance):
    try: 
        wallet = binance.fetch_balance(params={"type": "future"})
    except Exception as E:
        print(E)
        time.sleep(3)
        wallet = get_balance(binance)
    return wallet


def get_binance():
    with open("keys/a.txt") as f:
        lines = f.readlines()
        api_key = lines[0].strip()
        secret  = lines[1].strip()

    binance = ccxt.binance(config={
        'apiKey': api_key, 
        'secret': secret,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future'
        }
    })
    return binance

# 예쁜 print를 위해
def _b(str):
    return f"{Fore.BLUE}{str}{Style.RESET_ALL}"
def _r(str):
    return f"{Fore.RED}{str}{Style.RESET_ALL}"
def _y(str):
    return f"{Fore.YELLOW}{str}{Style.RESET_ALL}"
def _c(str):
    return f"{Fore.CYAN}{str}{Style.RESET_ALL}"
def _m(str):
    return f"{Fore.MAGENTA}{str}{Style.RESET_ALL}"

def pnlstr(pnlstr):
    if float(pnlstr) < -1:
        return _r(str(pnlstr)+"%")
    elif float(pnlstr) > 1:
        return _c(str(pnlstr)+"%")
    else:
        return str(pnlstr)+"%"
    
def status_str(status):
    if status == LONG:
        return _b(status)
    elif status == SHORT:
        return _m(status)
    else:
        return status

