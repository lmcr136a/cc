
import numpy as np
import pandas as pd
import time
import random
import matplotlib.pyplot as plt
from bull_bear import *
from datetime import datetime
from HYPERPARAMETERS import *



def select_sym(binance, tf, limit, wins, symnum):
    print(_y("\nSEARCHING..."))
    while 1:
        random.shuffle(SYMLIST)
        
        for sym in SYMLIST:  # 0705 0.55초 걸림
            short_only_strong, long_only_strong, befores = curr_states_other_minions(binance, ref_num=6)
            
            ms = get_ms(binance, sym, tf, limit, wins)
            timing_pos = timing_to_position(binance, ms, sym, tf, pr=True)
            
            timing = False
            
            if timing_pos:
                if (short_only_strong and timing_pos == LONG) or (long_only_strong and timing_pos == SHORT):
                    continue
                timing = True

            if timing:
                balance = get_balance(binance)
                positions = balance['info']['positions']
                for position in positions:
                    if position["symbol"] == sym.replace("/", ""):
                        amt = float(position['positionAmt'])
                        if amt == 0 and sym not in befores and sym.split("/")[0] not in ["MKR", "USDC", "ETC", "BNB", "BTC", "ETH", "BCH", "DASH", "XMR", "QNT", "LTC"]:
                            print(f"{sym} OOOOO")
                            return sym
            else:
                time.sleep(0.3*symnum)

                

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
