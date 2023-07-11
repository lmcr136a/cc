import datetime
import numpy as np
import pandas as pd
import mplfinance as mpf
import time
import argparse
import matplotlib.pyplot as plt
import ccxt 
from colorama import Fore, Style
from inspect_market import *

LONG = "Long"
SHORT = "Short"

with open("syms.txt", 'r') as f:
    SYMLIST = eval(f.read())
print(len(SYMLIST))

def cal_compound_amt(wallet_usdt, lev, price, symnum):
    return np.floor(wallet_usdt*lev/float(price)*0.9/float(symnum))


def select_sym(binance, __buying_cond, __pre_cond, tf, limit, wins):
    NEW_SYM = []
    while 1:
        for sym in SYMLIST:  # 0705 0.55초 걸림
            buying_cond, pre_cond = __buying_cond, __pre_cond

            actions = inspect_market(binance, sym, 1, buying_cond)
            short_only, long_only, buying_cond, _ = actions
        
            timing_pos = timing_to_position(binance, sym, buying_cond, pre_cond, tf, limit, wins, pr=False)
            
            timing = False
            if (timing_pos == SHORT and not long_only)\
                or (timing_pos == LONG and not short_only):
                timing = True

            if timing:
                balance = binance.fetch_balance()
                positions = balance['info']['positions']
                for position in positions:
                    if position["symbol"] == sym.replace("/", ""):
                        amt = float(position['positionAmt'])
                        if amt == 0 and "ETH" not in sym and "BCH" not in sym and "DASH" not in sym:
                            print(f"\n!\n!\n{sym} OOOOO")
                            return sym
            else:
                time.sleep(0.4)

        # with open("syms.txt", 'w') as f:
        #     f.write(str(NEW_SYM))
        # exit()

def get_ms(binance, sym, tf, limit, wins):
    df = past_data(binance, sym=sym, tf=tf, limit=limit)
    m1 = df['close'].rolling(window=wins[0]).mean()
    m2 =  df['close'].rolling(window=wins[1]).mean()
    m3 = df['close'].rolling(window=wins[2]).mean()
    m4 = df['close'].rolling(window=wins[3]).mean()
    return m1, m2, m3, m4


    
def get_curr_conds(binance, sym, tfs= ['1m', '3m', '5m', '30m'], limit=30):
    # limit=500이면 8.3시간, 24.9시간, 41.5시간, 10일
    # limit=180이면 3시간, 9시간, 15시간, 3일
    pos_val = []
    for tf in tfs:
        df = past_data(binance, sym, tf, limit=limit)
        m = df['close']
        v = get_curr_cond(m)
        pos_val.append(v)
    return pos_val


def get_curr_pnl(binance, sym):
    wallet = binance.fetch_balance(params={"type": "future"})
    positions = wallet['info']['positions']
    for pos in positions:
        if pos['symbol'] == sym:
            pnl = float(pos['unrealizedProfit'])/abs(float(pos['positionAmt']))/float(pos['entryPrice'])*100*float(pos['leverage'])
            return round(pnl,2), round(float(pos['unrealizedProfit']), 2)


def timing_to_close(binance, sym, status, curr_cond, does_m4_turnning, 
                    m1, satisfying_price, max_loss, min_profit, cond1, howmuchtime):
    curr_pnl, profit = get_curr_pnl(binance, sym.replace("/", ""))
    suddenly = isitsudden(m1, status)
    print(f"{sym} {howmuchtime} {status}] PRICE: {round(m1[-1], 2)} PNL: {profit} ({pnlstr(round(curr_pnl, 2))}), COND: {round(curr_cond, 2)} SAT_P: {satisfying_price}")
    
    if curr_pnl < max_loss \
        or\
    (
        curr_pnl > min_profit \
            and\
        does_m4_turnning
            and\
        (
            (curr_cond < -cond1 and status == SHORT) \
                or\
            (curr_cond > cond1 and status == LONG)
        )
    )\
        or\
    (suddenly and curr_pnl > satisfying_price):
        print(f"!!!{_y(sym)} {pnlstr(curr_pnl)} {status} {suddenly}")
        return True, curr_pnl
    else:
        return False, curr_pnl


def timing_to_position(binance, sym, buying_cond, pre_cond, tf, limit, wins, pr=True):
    m1, m2, m3 , m4 = get_ms(binance, sym, tf, limit, wins)
    turnning_shape = whether_turnning2(m2, m3, m4, ref=0.001*0.01, ref2=0.01*0.01)  # u or n or None
    val = get_curr_conds(binance, sym)
    # pre_cond = np.mean(val[1:])
    if pr:
        print(f'{sym} PRICE:', m1[-1], " SHAPE: ", turnning_shape, ' CONDS:', list(map(lambda x: round(x, 2), val)))

    if turnning_shape == 'u' and val[0] < -buying_cond:# and pre_cond < -pre_cond:
        return LONG
    elif turnning_shape == 'n' and val[0] > buying_cond:# and pre_cond > pre_cond:
        return SHORT
    else:
        return None



def isitsudden(m1, status, ref=0.08):
    now = m1[-1]
    prev = m1[-2]
    percent = (now - prev)/prev*100
    # print(percent, ref)
    if status == LONG and percent > ref:
        return True
    elif status == SHORT and percent < -ref:
        return True
    return False


def log_wallet_history(balance):
    wallet_info = np.load('wallet_log.npy')
    wallet_info = np.concatenate(
                        [wallet_info, 
                        [[time.time()], [float(balance['info']['totalWalletBalance'])]]],
                        axis=1
                        )  ## Date
    np.save('wallet_log.npy', wallet_info)
    plt.figure()
    plt.plot(wallet_info[0], wallet_info[1])
    plt.savefig("wallet_log.png")
    plt.close()


def minmax(m):
    m = (m - m.min())/(m.max() - m.min())
    return m



def whether_turnning2(m2, m3, m4, ref=0.001, ref2=0.002):
    d_m2 = np.diff(m2)      # rising?
    dd_m2 = np.diff(d_m2)   # concave?
    # ddd_m2 = np.diff(dd_m2) # curling up?
    d_m3 = np.diff(m3)      # rising?
    dd_m3 = np.diff(d_m3)   # concave?
    # ddd_m3 = np.diff(dd_m3) # curling up?
    
    n = m4_turn(LONG, m4)  # n
    u = m4_turn(SHORT, m4)  # u

    if u:
        return 'u'
    elif n:
        return 'n'
    return None


def get_curr_cond(m, period=500):
    m = m[-period:]
    mm = minmax(m)
    m_mean = np.mean(mm)
    return mm[-1] - m_mean
    
def show_total_pnl(transactions):
    pnls=[]
    for tr in transactions:
        pnls.append(float(tr['pnl'][:-1]))
    total_pnl = np.sum(pnls)
    return total_pnl


def m4_turn(status, m4, ref=0):
    i = -1
    m4_inc1 = m4[i-2] - m4[i-4] 
    m4_inc2 = m4[i] - m4[i-1] 
    if status == LONG: # n
        m4_turn = m4_inc1>ref and m4_inc2 <ref
    else:
        m4_turn = m4_inc1<ref and m4_inc2 >ref
    return m4_turn


def get_binance():
    with open("a.txt") as f:
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

def pnlstr(pnlstr):
    if float(pnlstr) < 0:
        return _r(str(pnlstr)+"%")
    elif float(pnlstr) > 0:
        return _c(str(pnlstr)+"%")
    else:
        return str(pnlstr)+"%"