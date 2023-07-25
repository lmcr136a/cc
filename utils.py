
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

init(convert=True)

def cal_compound_amt(wallet_usdt, lev, price, symnum):
    return np.floor(wallet_usdt*lev/float(price)*0.9/float(symnum))

"""
죽 돌면서 봤는데 다 겁나 상승 or 하강만 하느라 지그재그가 없음 => 90%이상 [[00001111], [11110000]] 페어
=> 지그재그더라도 내린게 더내리고 오른게 더오를 확률이 높음
=> 상승장에서 m3가 u모양이면 롱사기, m3가 n모양이면 숏사기

"""

def select_sym(binance, __buying_cond, __pre_cond, tf, limit, wins, symnum):
    print("SEARCHING...")
    while 1:
        for sym in SYMLIST:  # 0705 0.55초 걸림
            buying_cond, pre_cond = __buying_cond, __pre_cond
            actions = inspect_market(binance, sym, 1, buying_cond, print_=False)
            short_only, long_only, buying_cond, _ = actions
            
            ms = get_ms(binance, sym, tf, limit, wins)
            timing_pos = timing_to_position_score(binance, ms, sym, buying_cond, pre_cond, tf, limit, wins, pr=False)
            
            timing = False
            # if (timing_pos == SHORT and not long_only)\
            #     or (timing_pos == LONG and not short_only):
            if timing_pos:
                timing = True

            if timing:
                print(sym, "Timing")

                balance = binance.fetch_balance()
                positions = balance['info']['positions']
                for position in positions:
                    if position["symbol"] == sym.replace("/", ""):
                        amt = float(position['positionAmt'])
                        if amt == 0 and sym.split("/")[0] not in ["ETC", "BNB", "BTC", "ETH", "BCH", "DASH", "XMR", "QNT", "LTC"]:
                            print(f"\n!\n!\n{sym} OOOOO")
                            return sym
            else:
                time.sleep(0.2*symnum)
        # with open("syms.txt", 'w') as f:
        #     f.write(str(names))
        # exit()

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
    wallet = binance.fetch_balance(params={"type": "future"})
    positions = wallet['info']['positions']
    for pos in positions:
        if pos['symbol'] == sym:
            pnl = float(pos['unrealizedProfit'])/abs(float(pos['positionAmt']))/float(pos['entryPrice'])*100*float(pos['leverage'])
            return round(pnl,2), round(float(pos['unrealizedProfit']), 2)


def timing_to_close(binance, sym, status, m4_shape, 
                    ms, satisfying_price, max_loss, min_profit, buying_cond, howmuchtime, tf, limit, wins,):
    m1 = ms[0]
    curr_pnl, profit = get_curr_pnl(binance, sym.replace("/", ""))
    
    timing_pos = timing_to_position_score(binance, ms, sym, buying_cond, 0, tf, limit, wins, pr=False)
    if timing_pos:
        return False, curr_pnl

    suddenly = isitsudden(m1, status)
    if howmuchtime % 300 == 0 or (howmuchtime < 50 and howmuchtime % 10 ==0):
        print(f"{sym} {howmuchtime} {status_str(status)}] PNL: {profit} ({pnlstr(round(curr_pnl, 2))}), SAT_P: {satisfying_price}")
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


def timing_to_position_score(binance, ms, sym, buying_cond, pre_cond, tf, limit, wins, pr=True):
    # 더 점수가 높다의 뜻?

    m1, m2, m3 , m4 = ms

    curr_mvmt, curr_diff = curr_movement(m1, minute=5)  # 5개 시간봉의 움직임
    big_shape = np.diff(m4)[-3:]

    actions = inspect_market(binance, sym, 1, buying_cond, print_=False)
    short_only, long_only, buying_cond, satisfying_pnl = actions

    mm1 = minmax(m1[-int(4*60/float(tf[0])):])   # 4시간동안
    """
    그래프의 위치, 장의 흐름, 지금 당장의 움직임, 큰 흐름(m4)
    """
    if mm1[-1] > buying_cond and not short_only and curr_mvmt == FALLING \
        and np.all(big_shape > 0):
        print(mm1[-1], buying_cond, not short_only , curr_mvmt == FALLING, np.all(big_shape > 0) , 
            np.abs(curr_diff) < CALM*1.3, curr_diff, CALM)
        if np.abs(curr_diff) < CALM*1.3:
            return SHORT
        else:
            return LONG
    elif mm1[-1] < -buying_cond and not long_only and curr_mvmt == RISING \
        and np.all(big_shape < 0):
        if pr:
            print(mm1[-1], -buying_cond, not long_only , curr_mvmt == RISING, np.all(big_shape < 0) , 
                np.abs(curr_diff) < CALM*1.3, curr_diff, CALM)
        if np.abs(curr_diff) < CALM*1.3:
            return LONG
        else:
            return SHORT


    """
    이 아래로는 지그재그일때만 해당
    """
    zigzag, zzdic = handle_zigzag(m1, hour=4, tf=float(tf[0]))
    if not zigzag:
        return None
    if pr:
        print(sym, zzdic['where_h'])
        print(sym, zzdic['where_l']) 
    
    curr_mvmt, curr_diff = curr_movement(m1, minute=3)
    curr_diff = np.abs(curr_diff)
    
    # [큰 흐름] m3 (15개 이동평균선) 이 상승일때 롱, 하락이면 숏
    d_m3 = np.diff(m3)[-3:] # 두 번의 변화

    increasing_N_shortly_decreased = np.all(d_m3 > 0) and curr_mvmt == FALLING
    decreasing_N_shortly_increased = np.all(d_m3 < 0) and curr_mvmt == RISING
    
    """
    상승하다가 잠깐 하락? 하락하다가 잠깐 상승
    """
    if increasing_N_shortly_decreased or decreasing_N_shortly_increased:
        if mm1[-1] < -buying_cond:
            print(d_m3, curr_mvmt,  mm1[-1] , buying_cond)
            return SHORT
        elif mm1[-1] > buying_cond:
            print(d_m3, curr_mvmt,  mm1[-1] , -buying_cond)
            return LONG
        

# def timing_to_position(binance, sym, buying_cond, pre_cond, tf, limit, wins, pr=True):
#     m1, m2, m3 , m4 = get_ms(binance, sym, tf, limit, wins)
#     turnning_shape = m4_turn(m4)
    
#     curr_mvmt, last_diff = curr_movement(m1)  # 2개 시간봉의 움직임
#     last_diff = np.abs(last_diff)
#     if last_diff > CALM:   # 위험
#         return None
#     # pre_cond = np.mean(val[1:])
#     if pr:
#         print(f'{sym} PRICE:', m1[-1], " SHAPE: ", turnning_shape, curr_mvmt)
#     actions = inspect_market(binance, sym, 1, buying_cond, print_=False)
#     short_only, long_only, buying_cond, _ = actions

#     # [큰 흐름] m3 (15개 이동평균선) 이 상승일때 롱, 하락이면 숏
#     d_m3 = np.diff(m2)[-3:] # 두 번의 변화

#     # [작은 흐름] 순간의 급락: mvmt
#     increasing_N_shortly_decreased = np.all(d_m3 > 0) and curr_mvmt == FALLING
#     decreasing_N_shortly_increased = np.all(d_m3 < 0) and curr_mvmt == RISING

#     mm1 = minmax(m1)
#     # print(np.all(d_m3 > 0),curr_mvmt == FALLING, not short_only)
#     # print(np.all(d_m3 < 0), curr_mvmt == RISING, not long_only)
#     if increasing_N_shortly_decreased and not short_only:
#         return LONG
#     elif decreasing_N_shortly_increased and not long_only:
#         return SHORT
    
#     if mm1[-1] < -buying_cond and not short_only:
#         return LONG
#     elif mm1[-1] > buying_cond and not long_only:
#         return SHORT

    # 애네는 급등 급락시임 아 근데 if else로 하기엔 너무 복잡하다
    # if mvmt == RISING and last_diff > CALM*10:

    #     return SHORT
    # elif mvmt == FALLING and last_diff > CALM*10:
    #     return LONG
    # else:
    #     return None


def curr_movement(m, minute=2):
    diff = []
    m = m[-minute-1:]
    for i in range(len(m)-1):
        diff.append(m[i+1] - m[i])
    d = np.sum(diff)

    diff = (m[-1] - m[0])/m[-2]*100
    last_diff = (m[-1] - m[0])/m[-2]
    if d > 0 and (last_diff > 0):
        return RISING, diff
    elif d < 0 and (last_diff < 0):
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
    plt.plot(wallet_info[0], wallet_info[2], 'b--')
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
    if float(pnlstr) < 0:
        return _r(str(pnlstr)+"%")
    elif float(pnlstr) > 0:
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