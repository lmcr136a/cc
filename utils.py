
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
# from get_model import get_model_prediction
from ccxt.base.errors import BadSymbol
import asyncio
import ccxt.pro as ccxtpro
# init(convert=True)

# def cal_compound_amt(wallet_usdt, lev, price, symnum):
#     return float(wallet_usdt*lev/float(price)*0.9/float(symnum))
def cal_compound_amt(wallet_usdt, lev, price, symnum):
    return float(60*lev/float(price)*0.98)

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
    if n1 > 3:
        short_only_strong = True
        # print("Short Only..")
    if n2 > 3:
        long_only_strong = True
        # print("Long Only...")
    return short_only_strong, long_only_strong, befores


async def select_sym(symnum):
    binance = get_binance()
    print(_y("\nSEARCHING..."))
    max_score, min_score = 0,0
    max_sym, min_sym = 0,0
    return_pos = None
    
    market_dic = {
        "//":[],
        "-/":[],
        "--":[],
        "-d":[],
        "dd":[],
        "d-":[],
        "/-":[],
        "v":[],
        "^":[],
    }
    
    while return_pos is None:
        random.shuffle(SYMLIST)
        for i, sym in enumerate(SYMLIST):  # 0705 0.55초 걸림
            vol = await binance.fetch_tickers(symbols=[sym])
            time.sleep(0.5)
            await binance.close()
                
            if (not len(list(vol.values())) > 0) or list(vol.values())[0]['quoteVolume'] < 20*(10**6):
                continue
            
            print(f"[{i}/{len(SYMLIST)}] ", end="")

            try:
                shape, score = await inspect_market(binance, sym, print_=True)
                if shape:
                    market_dic[shape].append([sym, score])
                
            except BadSymbol as E:
                SYMLIST.pop(SYMLIST.index(sym))
                print(f"REMOVE {sym} from DB")
                with open("symlist.txt", "w") as f:
                    f.write(str(SYMLIST))
                continue
            
            if i > 20:
                break
            
        
        shape_nums = list(map(lambda x: len(x), market_dic.values()))
        common_shape = list(market_dic.keys())[np.argmax(shape_nums)]
        if common_shape in ["--", "^", "//", "/-", '-/']:
            return_pos = LONG
        elif common_shape in ["v", 'dd', '-d','d-']:
            return_pos = SHORT
        if len(market_dic[common_shape]) > 0:
            print(market_dic[common_shape], common_shape)
            common_shape_scores = np.array(market_dic[common_shape])[:, 1].astype(np.float32)
            
            return_sym = np.array(market_dic[common_shape])[:,0][np.argmax(np.abs(common_shape_scores))]
            print(f" common shape: {common_shape} <<{return_sym}>>")
            await binance.close()
            return return_sym ,return_pos

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
    
async def get_curr_pnl(sym):
    binance = get_binance()
    await binance.fetch_markets()
    balance = await binance.fetch_balance(params={"type": "future"})
    await binance.close()
    positions = balance['info']['positions']
    
    pnl, profit = 0,0
    for position in positions:
        if position['symbol'] == sym.replace("/", ""):
            pose_info = position
            pnl = float(pose_info['unrealizedProfit'])/float(pose_info['initialMargin'])*100
            profit = pose_info['unrealizedProfit']
            
    return round(pnl,2), round(float(profit), 2)


def timing_to_close(sym, satisfying_profit, max_loss):
    curr_pnl, profit = asyncio.run(get_curr_pnl(sym.replace("/", "")))
    # sat_cond = curr_pnl > satisfying_profit
    # if sat_cond:
    #     print(f"\n!!! Satisfied: {curr_pnl}")
    #     return True, curr_pnl

    # timing_pos = timing_to_position_score(binance, ms, sym, buying_cond, 0, tf, limit, wins, pr=False)
    print(f"\r{sym.split('/')[0]} ] PNL: {profit} ({pnlstr(round(curr_pnl, 1))})|{satisfying_profit}%\t", end="")
    if curr_pnl < max_loss:
        print(f"\n!!! max_loss: {curr_pnl}")
        return True, curr_pnl

    return False, curr_pnl

def timing_to_position(binance, ms, sym, buying_cond, pre_cond, tf, limit, wins, pr=True):
    # model_pred = timing_to_position_model(binance, sym)
    rule_pred = short_mvmt_timing(binance, ms, sym, buying_cond, pre_cond, tf, limit, wins, pr=pr)
    if rule_pred:
        return rule_pred

# def timing_to_position_model(binance, sym):
#     return get_model_prediction(binance=binance, sym=sym)


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


def short_mvmt_timing(binance, ms, sym, buying_cond, pre_cond, tf, limit, wins, pr=True):
    # 더 점수가 높다의 뜻?

    m1, m2, m3 , m4 = ms

    curr_mvmt, amount = curr_movement(m1, minute=5)  # 5개 시간봉의 움직임
    # small_shape = shape_info(m2)
    ref = 0.9
    
    if pr:
        print(f"[{sym[:-5]}] curr_mvmt:{curr_mvmt} amt: {amount}")
        
    if curr_mvmt == FALLING:
        if amount < -ref:
            return SHORT
        elif amount > -ref:
            return LONG   
    elif curr_mvmt == RISING:
        if amount > ref:
            return LONG
        elif amount < ref:
            return SHORT
    

    # """
    # 이 아래로는 지그재그일때만 해당
    # """
    # zigzag, zzdic = handle_zigzag(m1, hour=4, tf=float(tf[0]))
    # if not zigzag:
    #     return None
    # if pr:
    #     print(sym, zzdic['where_h'])
    #     print(sym, zzdic['where_l']) 
    
    # curr_mvmt, curr_diff = curr_movement(m1, minute=3)
    # curr_diff = np.abs(curr_diff)
    
    # # [큰 흐름] m3 (15개 이동평균선) 이 상승일때 롱, 하락이면 숏
    # i = int(round(4*60/float(tf[0])))
    # d_m3 = np.diff(m3)[-3:] # 두 번의 변화
    # d_m4 = np.diff(m4[-i:]) # 4시간동안의 변화
    # m4inc = np.sum(np.where(d_m4 > 0, 1, 0))/len(d_m4) > 0.8
    # m4dec = np.sum(np.where(d_m4 < 0, 1, 0))/len(d_m4) > 0.8
    
    # increasing_N_shortly_decreased = np.all(d_m3 > 0) and curr_mvmt == FALLING
    # decreasing_N_shortly_increased = np.all(d_m3 < 0) and curr_mvmt == RISING
    
    # """
    # 상승하다가 잠깐 하락? 하락하다가 잠깐 상승
    # """
    # if increasing_N_shortly_decreased or decreasing_N_shortly_increased:
    #     if mm1[-1] < -buying_cond:
    #         if pr:
    #             print("[CASE3] ", d_m3, curr_mvmt,  mm1[-1], -buying_cond, m4dec)
    #         if m4dec:
    #             return SHORT
    #         else:
    #             return LONG
    #     elif mm1[-1] > buying_cond:
    #         if pr:
    #             print("[CASE4] ", d_m3, curr_mvmt,  mm1[-1] , buying_cond, m4inc)
    #         if m4inc:
    #             return LONG
    #         else:
    #             return SHORT
        
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
        diff.append((m[i+1] - m[i])/m[i])
    d = np.sum(diff)*100

    if d > ref*minute/2:
        return RISING, d
    elif d < -ref*minute/2:
        return FALLING, d
    else:
        return "~", d


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

    binance = ccxtpro.binance(config={
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