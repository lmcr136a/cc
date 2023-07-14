
import numpy as np
import pandas as pd
import mplfinance as mpf
import time
import argparse
import matplotlib.pyplot as plt
import ccxt 
from colorama import Fore, Style, init
from inspect_market import *
from datetime import datetime

init(convert=True)
RISING = "Rising"
FALLING = "Falling"
LONG = "Long"
SHORT = "Short"

CALM = 0.05 # 20배일때 1%

with open("symlist.txt", 'r') as f:
    SYMLIST = eval(f.read())
print(len(SYMLIST))

def cal_compound_amt(wallet_usdt, lev, price, symnum):
    return np.floor(wallet_usdt*lev/float(price)*0.9/float(symnum))


def select_sym(binance, __buying_cond, __pre_cond, tf, limit, wins, symnum):
    print("SEARCHING...")
    while 1:
        for sym in SYMLIST:  # 0705 0.55초 걸림
            buying_cond, pre_cond = __buying_cond, __pre_cond
            actions = inspect_market(binance, sym, 1, buying_cond, print_=False)
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
                        if amt == 0 and sym.split("/")[0] not in ["BTC", "ETH", "BCH", "DASH", "XMR", "QNT", "LTC"]:
                            print(f"\n!\n!\n{sym} OOOOO")
                            return sym
            else:
                time.sleep(0.2*symnum)

        # with open("syms.txt", 'w') as f:
        #     f.write(str(names))
        # exit()

def get_ms(binance, sym, tf, limit, wins):
    df = past_data(binance, sym=sym, tf=tf, limit=limit)
    m1 = df['close'].rolling(window=wins[0]).mean()
    m2 =  df['close'].rolling(window=wins[1]).mean()
    m3 = df['close'].rolling(window=wins[2]).mean()
    m4 = df['close'].rolling(window=wins[3]).mean()
    return m1, m2, m3, m4


def get_curr_pnl(binance, sym):
    wallet = binance.fetch_balance(params={"type": "future"})
    positions = wallet['info']['positions']
    for pos in positions:
        if pos['symbol'] == sym:
            pnl = float(pos['unrealizedProfit'])/abs(float(pos['positionAmt']))/float(pos['entryPrice'])*100*float(pos['leverage'])
            return round(pnl,2), round(float(pos['unrealizedProfit']), 2)


def timing_to_close(binance, sym, status, m4_shape, 
                    m1, satisfying_price, max_loss, min_profit, cond1, howmuchtime):
    curr_pnl, profit = get_curr_pnl(binance, sym.replace("/", ""))
    suddenly = isitsudden(m1, status)
    if howmuchtime % 300 == 0:
        print(f"{sym} {howmuchtime} {status}] PNL: {profit} ({pnlstr(round(curr_pnl, 2))}), SAT_P: {satisfying_price}")
    mvmt, last_diff = curr_movement(m1, minute=4)
    # 이 이상 잃을 수는 없다
    loss_cond = curr_pnl < max_loss

    # 지그재그에서 위쪽이면 롱 팔고 아래면 숏 팔고
    zz_cond = False
    zigzag, zzdic = handle_zigzag(m1)
    if zigzag:
        if (status == LONG and m1[-1] > zzdic['zzmax']) or\
        (status == SHORT and m1[-1] < zzdic['zzmin']):
            zz_cond = True

    # u 또는 n
    shape_cond = (((m4_shape=='u' and mvmt==FALLING and last_diff < CALM and status == SHORT) \
                    or\
                    (m4_shape=='n' and mvmt==RISING and last_diff < CALM and status == LONG)))
    
    # 적당히 먹었다!
    sat_cond = not suddenly and curr_pnl > satisfying_price

    if loss_cond or sat_cond or ((zz_cond or shape_cond) and curr_pnl > min_profit):
        print(f"!!!{_y(sym)} {pnlstr(curr_pnl)} {loss_cond} {shape_cond} {sat_cond} {zz_cond}")
        return True, curr_pnl
    else:
        return False, curr_pnl


def timing_to_position(binance, sym, buying_cond, pre_cond, tf, limit, wins, pr=True):
    m1, m2, m3 , m4 = get_ms(binance, sym, tf, limit, wins)
    turnning_shape = m4_turn(m4)
    
    mvmt, last_diff = curr_movement(m1)
    last_diff = np.abs(last_diff)
    # pre_cond = np.mean(val[1:])
    if pr:
        print(f'{sym} PRICE:', m1[-1], " SHAPE: ", turnning_shape, mvmt)
    actions = inspect_market(binance, sym, 1, buying_cond, print_=False)
    short_only, long_only, buying_cond, _ = actions

    # 오른게 더오르고 내린게 더내려간다
    # 그냥 약간 올랐을때로 변경
    if mvmt == RISING and last_diff < CALM and not short_only:
        return LONG
    elif mvmt == FALLING and last_diff < CALM and not long_only:
        return SHORT
    
    # 애네는 급등 급락시임 아 근데 if else로 하기엔 너무 복잡하다
    # if mvmt == RISING and last_diff > CALM*10:

    #     return SHORT
    # elif mvmt == FALLING and last_diff > CALM*10:
    #     return LONG
    else:
        return None


def curr_movement(m, minute=2):
    diff = []
    for i in range(len(m)-1):
        diff.append(m[i+1] - m[i])
    d = np.sum(diff)

    last_diff = (m[-1] - m[-2])/m[-2]
    if d > 0 and ( last_diff > 0):
        return RISING, last_diff
    elif d < 0 and (last_diff < 0):
        return FALLING, last_diff
    else:
        return "~", last_diff


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


def log_wallet_history(balance):
    today = datetime.now()
    try:
        wallet_info = np.load('wallet_log.npy')
        wallet_info = np.concatenate(
                            [wallet_info, 
                            [[today.strftime('%y/%m/%d-%H:%M')], [float(balance['info']['totalWalletBalance'])]]],
                            axis=1
                            )  ## Date
    except FileNotFoundError:
        wallet_info = np.array([[time.time()], [float(balance['info']['totalWalletBalance'])]])
    np.save('wallet_log.npy', wallet_info)
    plt.figure()
    plt.plot(wallet_info[0], wallet_info[1])
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
def handle_zigzag(m1, hour=2):
    # 1분봉으로 2시간동안 0.9*max와 0.9*min에 몇번 도달했는지?
    # 기준선들 각 2번 이상씩 찍으면 지그재그  (1,1)이면 상승 또는 하강, (2,1)이면 u또는 n
    m1 = m1[-hour*60:]
    his_2h = minmax(m1)
    ref_h, ref_l = 0.8, -0.8

    where_h = np.where(his_2h>ref_h, 1, 0).reshape(-1, 2)                 # where_h: (60, 2)
    where_l = np.where(his_2h<ref_l, 1, 0).reshape(-1, 2)

    where_h = np.where(np.sum(where_h, axis=1) > 0, 1, 0).reshape(-1, 6)  # where_h: (60) -> (10, 6)
    where_l = np.where(np.sum(where_l, axis=1) > 0, 1, 0).reshape(-1, 6)  # 연속적인거 카운트 안하기 위해

    where_h = np.where(np.sum(where_h, axis=1) > 0, 1, 0)                 # where_h: (10)
    where_l = np.where(np.sum(where_l, axis=1) > 0, 1, 0)

    h_num, l_num = 0, 0
    for i in range(len(where_h)-1):
        h = [where_h[i], where_h[i+1]]
        if h == [0, 1] or h == [1, 0]:
            h_num += 0.5
        l = [where_l[i], where_l[i+1]]
        if l == [0, 1] or l == [1, 0]:
            l_num += 0.5
    # print(where_h, h_num)
    # print(where_l, l_num)
    if h_num >= 2 and l_num >= 2 and (h_num+l_num) >= 5:
        return True, {"zzmin":inv_minmax(m1, -(ref_l-0.2)), "zzmax":inv_minmax(m1, ref_h-0.2)}
    return False, {}


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

def pnlstr(pnlstr):
    if float(pnlstr) < 0:
        return _r(str(pnlstr)+"%")
    elif float(pnlstr) > 0:
        return _c(str(pnlstr)+"%")
    else:
        return str(pnlstr)+"%"