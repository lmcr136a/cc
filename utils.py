import datetime
import numpy as np
import pandas as pd
import mplfinance as mpf
import time
import argparse
import matplotlib.pyplot as plt
import ccxt 

LONG = "Long"
SHORT = "Short"

with open("syms.txt", 'r') as f:
    SYMLIST = eval(f.read())
print(len(SYMLIST))

def cal_compound_amt(wallet_usdt, lev, price, symnum):
    return np.floor(wallet_usdt*lev/price*0.9/symnum)


def select_sym(binance, buying_cond, pre_cond, tf, limit, wins):
    NEW_SYM = []
    while 1:
        for sym in SYMLIST:
            try:
                timing = timing_to_position(binance, sym, buying_cond, pre_cond, tf, limit, wins, pr=False)
                NEW_SYM.append(sym)
            except:
                pass
            if timing:
                balance = binance.fetch_balance()
                positions = balance['info']['positions']
                for position in positions:
                    if position["symbol"] == sym.replace("/", ""):
                        amt = float(position['positionAmt'])
                        if amt == 0:
                            print(f"{sym} O")
                            return sym
            else:
                print(f"{sym} X")

    with open("syms.txt", 'w') as f:
        f.write(str(NEW_SYM))

def get_ms(binance, sym, tf, limit, wins):
    df = past_data(binance, sym=sym, tf=tf, limit=limit)
    m1 = df['close'].rolling(window=wins[0]).mean()
    m2 =  df['close'].rolling(window=wins[1]).mean()
    m3 = df['close'].rolling(window=wins[2]).mean()
    m4 = df['close'].rolling(window=wins[3]).mean()
    return m1, m2, m3, m4


def past_data(binance, sym, tf, limit, since=None):
    coininfo = binance.fetch_ohlcv(
        symbol=sym, 
        timeframe=tf, 
        since=since, 
        limit=limit
    )
    df = pd.DataFrame(coininfo, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
    df.set_index('datetime', inplace=True)
    return df

    
def get_curr_conds(binance, sym, tfs= ['1m', '3m', '5m', '30m'], limit=180):
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
    print(f"{sym} {howmuchtime}] PRICE: {m1[-1]} PNL: {profit} ({round(curr_pnl, 2)}%), COND: {round(curr_cond, 2)} SAT_P: {satisfying_price}")
    
    if curr_pnl < max_loss \
        or\
    (
        curr_pnl > min_profit \
            and\
        does_m4_turnning
            and\
        (
            (curr_cond < cond1 and status == SHORT) \
                or\
            (curr_cond > cond1 and status == LONG)
        )
    )\
        or\
    (suddenly and curr_pnl > satisfying_price):
        print(f"!!!{sym}")
        print(curr_pnl, status, suddenly)
        return curr_pnl
    else:
        return False


def timing_to_position(binance, sym, buying_cond, pre_cond, tf, limit, wins, pr=True):
    m1, m2, m3 , m4 = get_ms(binance, sym, tf, limit, wins)
    turnning_shape = whether_turnning2(m2, m3, m4, ref=0.001*0.01, ref2=0.01*0.01)  # u or n or None
    val = get_curr_conds(binance, sym)
    pre_cond = np.mean(val[1:])
    if pr:
        print(f'{sym} PRICE:', m1[-1], " SHAPE: ", turnning_shape, ' CONDS:', list(map(lambda x: round(x, 2), val)))

    if turnning_shape == 'u' and val[0] < -buying_cond and pre_cond < -pre_cond:
        return LONG
    elif turnning_shape == 'n' and val[0] > buying_cond and pre_cond > pre_cond:
        return SHORT
    else:
        return None



def isitsudden(m1, status, ref=0.08):
    now = m1[-1]
    prev = m1[-2]
    percent = (now - prev)/prev*100
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

# TODO: 
# 심볼 탐색후 선택


def bull_or_bear(df):  # 상승장인지 하락장인지?
    # df: 2H time frame, 3일간의 흐름을 본다! 길이 12*3
    tfnum = 24/2  # 하루의 막대개수
    date = 3
    m = df['close'][-int(tfnum*date):]
    blocks = [0, int(tfnum*date-tfnum), int(tfnum*date-0.5*tfnum)]
    rising = []
    for b in blocks:
        for i in range(1, len(m)-b):
            rising.append((m[b+i] - m[b])/m[b]*100)
    rising_coef = np.mean(rising)
    if rising_coef > 2:
        return "BULL"
    elif rising_coef < -1.5:
        return "BEAR"
    else:
        return "~-~-~"
    



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
    
    # m2 m3 crossed
    prevs = [-3, -4, -5]
    diff = m2[prevs] - m3[prevs]
    prev_m2_undr_m3 = np.all(diff < 0)
    prev_m2_on_m3 = np.all(diff > 0)

    curr_diff = m2[-1] - m3[-1]
    curr_m2_on_m3 = curr_diff > 0
    curr_m2_undr_m3 = curr_diff < 0

    r = 3
    conref = 0.0009
    conref2 = 0.0007
    m2_concave = np.mean(dd_m2[-r:]) > conref
    m3_concave = np.mean(dd_m3[-r:]) > conref2
    hueck = dd_m2[-1] > conref

    m2_convex = np.mean(dd_m2[-r:]) < -conref
    m3_convex = np.mean(dd_m3[-r:]) < -conref2
    hueck_ = dd_m2[-1] < -conref

    concave = m2_concave and m3_concave and hueck
    convex = m2_convex and m3_convex and hueck_
    str = ''
    print_ = True
    if print_:
        if convex:
            str += '볼록'
        if concave:
            str += '오목'
        if curr_m2_on_m3:
            str += ' m2가 위에'
        if curr_m2_undr_m3:
            str += ' m3가 위에'

        str += " | m2 "
        if m2_convex:
            str += '볼록'
        if m2_concave:
            str += '오목'
        if hueck or hueck_:
            str += '획'

        str += " | m3 "
        if m3_convex:
            str += '볼록'
        if m3_concave:
            str += '오목'

    if prev_m2_undr_m3 and curr_m2_on_m3 and concave:
        return 'u'
    elif prev_m2_on_m3 and curr_m2_undr_m3 and convex:
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


def m4_turn(status, m2, m3, m4, ref=0):
    i = -1
    m4_inc1 = m4[i-2] - m4[i-4] 
    m4_inc2 = m4[i] - m4[i-1] 
    if status == LONG: # n
        m4_turn = m4_inc1>ref and m4_inc2 <ref
    else:
        m4_turn = m4_inc1<ref and m4_inc2 >ref
    return m4_turn

def close(pos, tr, profit, lev=1):
    p=1
    if pos == SHORT:
        p = -1
    tr['position'] = pos
    tr['pnl'] = str(round(profit, 4))+"%"
    return tr


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

