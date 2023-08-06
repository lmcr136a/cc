
import numpy as np
import pandas as pd
import mplfinance as mpf
import time
import random
import matplotlib.pyplot as plt
import ccxt 
from colorama import Fore, Style, init
from bull_bear import *
from datetime import datetime
from HYPERPARAMETERS import *
from get_model import get_model_prediction

# init(convert=True)

def cal_compound_amt(wallet_usdt, lev, price, symnum):
    return np.float(wallet_usdt*lev/float(price)*0.9/float(symnum))

"""
죽 돌면서 봤는데 다 겁나 상승 or 하강만 하느라 지그재그가 없음 => 90%이상 [[00001111], [11110000]] 페어
=> 지그재그더라도 내린게 더내리고 오른게 더오를 확률이 높음
=> 상승장에서 m3가 u모양이면 롱사기, m3가 n모양이면 숏사기

"""

def curr_states_other_minions():
    with open("before_sym.txt", 'r') as f:
        befores = f.read()
    befores = befores.split("\n")
    # print(befores)
    positions = get_existing_positions()
    n1, n2 = 0, 0
    for pos in positions:
        if pos == LONG:
            n1 +=1
        elif pos == SHORT:
            n2 += 1
    short_only_strong, long_only_strong = False, False
    if n1 > 7:
        short_only_strong = True
        # print("Short Only..")
    if n2 > 7:
        long_only_strong = True
        # print("Long Only...")
    return short_only_strong, long_only_strong, befores


def select_sym(binance, tf, limit, wins, symnum):
    print(_y("\nSEARCHING..."))
    while 1:
        random.shuffle(SYMLIST)
        
        for sym in SYMLIST:  # 0705 0.55초 걸림
            short_only_strong, long_only_strong, befores = curr_states_other_minions()
            
            ms = get_ms(binance, sym, tf, limit, wins)
            timing_pos = timing_to_position(binance, ms, sym, tf, pr=True)
            
            timing = False
            
            if timing_pos:
                if (short_only_strong and timing_pos == LONG) or (long_only_strong and timing_pos == SHORT):
                    continue
                timing = True

            if timing:
                balance = binance.fetch_balance()
                positions = balance['info']['positions']
                for position in positions:
                    if position["symbol"] == sym.replace("/", ""):
                        amt = float(position['positionAmt'])
                        if amt == 0 and sym not in befores and sym.split("/")[0] not in ["MKR", "USDC", "ETC", "BNB", "BTC", "ETH", "BCH", "DASH", "XMR", "QNT", "LTC"]:
                            print(f"{sym} OOOOO")
                            return sym
            else:
                time.sleep(0.3*symnum)
        # with open("syms.txt", 'w') as f:
        #     f.write(str(names))
        # exit()


def timing_to_position(binance, ms, sym, tf, pr=True):
    tf_ = int(tf.replace("m", ""))
    m1, m2, m3 , m4 = ms

    curr_mvmt, curr_diff = curr_movement(m1, minute=5)  # 5개 시간봉의 움직임
    big_shape = np.diff(m4)[-3:]
    small_shape = shape_info(m2)

    # actions = inspect_market(binance, sym, 1, buying_cond, print_=False)
    # short_only, long_only, buying_cond, satisfying_pnl = actions

    """
    그래프의 위치, 장의 흐름, 지금 당장의 움직임, 큰 흐름(m4)
    """
    if curr_mvmt == FALLING:
        if pr:
            print(f"[{sym[:-5]} CASE1] curr_mvmt:{curr_mvmt} {small_shape}")
        # if small_shape == INCREASING_CONCAVE: 
        #     return long_cond(m1, tf_=tf_)
        if small_shape == DECREASING_CONVEX:
            return short_cond(m1, tf_=tf_)
        
    elif curr_mvmt == RISING:
        if pr:
            print(f"[{sym[:-5]} CASE2] curr_mvmt:{curr_mvmt} {small_shape}")
        if small_shape == INCREASING_CONCAVE:  
            return long_cond(m1, tf_=tf_)
        # if small_shape == DECREASING_CONVEX:
        #     return short_cond(m1, tf_=tf_)
        
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

def whether_calm(m1, ref=0.05, n=80):
    m = m1[-n:]
    li = []
    for i in range(n-1):
        pre = m[-(i+1)]
        now = m[-i]
        li.append(np.abs(now-pre)/pre*100)

    print(np.mean(li), np.max(li), np.std(li))
    if np.std(li) <= ref:
        return True
    elif np.std(li) > ref:
        return False
    
def get_curr_pnl(binance, sym):
    try:
        wallet = binance.fetch_balance(params={"type": "future"})
    except Exception as E:
        print(E)
        time.sleep(3)
        wallet = binance.fetch_balance(params={"type": "future"})
    
    positions = wallet['info']['positions']
    for pos in positions:
        if pos['symbol'] == sym:
            try:
                pnl = float(pos['unrealizedProfit'])/abs(float(pos['positionAmt']))/float(pos['entryPrice'])*100*float(pos['leverage'])
            except:
                print(float(pos['unrealizedProfit']), abs(float(pos['positionAmt'])), float(pos['entryPrice']), float(pos['leverage']))
                pnl = 0
            return round(pnl,2), round(float(pos['unrealizedProfit']), 2)

def isit_wrong_position(m3, status, n=3):
    # m3 은 상승(하락)하는데 SHORT(LONG) 포지션이다?!
    d_m3 = np.diff(m3[-10:])[-n:]

    if (np.all(d_m3 > 0) and status == SHORT) or\
        (np.all(d_m3 < 0) and status == LONG):
        return True
    return False


def log_wallet_history(balance):
    today = datetime.now()
    try:
        wallet_info = np.load('wallet_log.npy')
        wallet_info = np.concatenate(
                            [wallet_info, 
                            [[time.time()], 
                            [float(balance['info']['totalWalletBalance'])],
                            [float(balance['info']['totalMarginBalance'])] ]],
                            axis=1
                            )  ## Date
    except FileNotFoundError:
        wallet_info = np.array([[time.time()], [float(balance['info']['totalWalletBalance'])], [float(balance['info']['totalMarginBalance'])]])
    np.save('wallet_log.npy', wallet_info)
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

# 지그재그인지 확인, w 또는 m, n&u&un은 안됨
def handle_zigzag(m1, hour=2, tf=1):
    # tf분봉으로 hour시간동안 0.9*max와 0.9*min에 몇번 도달했는지?
    # 기준선들 각 2번 이상씩 찍으면 지그재그  (1,1)이면 상승 또는 하강, (2,1)이면 u또는 n
    m1 = m1[-int(hour*60/tf):]
    his_2h = minmax(m1)
    ref_h, ref_l = 0.65, -0.65

    where_h = np.where(his_2h>ref_h, 1, 0).reshape(-1, 2)                 # where_h: (40, 2)
    where_l = np.where(his_2h<ref_l, 1, 0).reshape(-1, 2)

    where_h = np.where(np.sum(where_h, axis=1) > 0, 1, 0).reshape(-1, 5)  # where_h: (40) -> (8, 5)
    where_l = np.where(np.sum(where_l, axis=1) > 0, 1, 0).reshape(-1, 5)  # 연속적인거 카운트 안하기 위해

    where_h = np.where(np.sum(where_h, axis=1) > 0, 1, 0)                 # where_h: (8)
    where_l = np.where(np.sum(where_l, axis=1) > 0, 1, 0)

    h_num, l_num = 0, 0
    for i in range(len(where_h)-1):
        h = [where_h[i], where_h[i+1]]
        if h == [0, 1] or h == [1, 0]:
            h_num += 0.5
        l = [where_l[i], where_l[i+1]]
        if l == [0, 1] or l == [1, 0]:
            l_num += 0.5
    # ####
    # if len(where_h) < 10:
    #     print(where_h, h_num)
    #     print(where_l, l_num)
    # ####
    
    l = round(len(where_h)/2)
    # print(np.sum(where_h[:l])*np.sum(where_l[:l])*np.sum(where_h[l:])*np.sum(where_l[l:]) )
    if h_num >= 1.5 and l_num >= 1.5 and (np.sum(where_h[:l])*np.sum(where_l[:l])*np.sum(where_h[l:])*np.sum(where_l[l:]) > 0):
        return True, {"zzmin":inv_minmax(m1, -(ref_l-0.2)), "zzmax":inv_minmax(m1, ref_h-0.2),
                      "where_h":where_h, "where_l": where_l}
    if h_num >= 1 and l_num >= 1 and (h_num+l_num >= 4):
        return True, {"zzmin":inv_minmax(m1, -(ref_l-0.2)), "zzmax":inv_minmax(m1, ref_h-0.2),
                      "where_h":where_h, "where_l": where_l}
    return False, {}


def get_existing_positions():
    with open("positions.txt", 'r') as f:
        positions = f.read()
    positions = eval(positions)
    return positions

def pop_from_existing_positions(status):
    positions = get_existing_positions()
    positions.pop(positions.index(status))
    with open('positions.txt', 'w') as f:
        f.write(str(positions))

def add_to_existing_positions(status):
    positions = get_existing_positions()
    positions.append(status)
    with open('positions.txt', 'w') as f:
        f.write(str(positions))


# def get_curr_cond(m, period=500):
#     m = m[-period:]
#     mm = minmax(m)
#     return mm[-1]
    
def show_total_pnl(transactions):
    pnls=[]
    for tr in transactions:
        pnls.append(float(tr['pnl'][:-1]))
    total_pnl = np.sum(pnls)
    return total_pnl


def m4_turn(m4, ref=0.005):
    i = -1
    m4_inc1 = (m4[i-3] - m4[i-5])/m4[i-5]*100
    m4_inc2 = (m4[i-4] - m4[i-6])/m4[i-6]*100
    m4_inc3 = (m4[i-5] - m4[i-7])/m4[i-7]*100

    m4_dec_now1 = (m4[i] - m4[i-1])/m4[i-1]*100
    m4_dec_now2 = (m4[i-1] - m4[i-2])/m4[i-2]*100

    # n
    m4_increased = m4_inc1 >0 and m4_inc2 >ref and m4_inc3 >ref
    if m4_increased and m4_dec_now1 < 0 and m4_dec_now2 < -ref:
        return 'n'
    # u
    m4_decreased = m4_inc1 <0 and m4_inc2 < -ref and m4_inc3 < -ref
    if m4_decreased and m4_dec_now1 > 0 and m4_dec_now2 > ref:
        return 'u'


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

