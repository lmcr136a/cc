
import numpy as np
import pandas as pd
import time
import random
import matplotlib.pyplot as plt
import ccxt 
from colorama import Fore, Style, init
from datetime import datetime
from HYPERPARAMETERS import *
import asyncio
import ccxt.pro as ccxtpro
import matplotlib.dates as mdates

# def cal_compound_amt(wallet_usdt, lev, price, symnum):
#     return float(wallet_usdt*lev/float(price)*0.9/float(symnum))
def cal_compound_amt(lev, price):
    return float(5*lev/float(price))

async def retry_on_error(func, *args, **kwargs):
    max_retries = 3
    retry_delay = 3
    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as e:
            if attempt == max_retries - 1:
                print(f"최대 재시도 횟수 초과: {e}")
                raise
            print(f"통신 에러 발생 (시도 {attempt + 1}/{max_retries}): {e}")
            await asyncio.sleep(retry_delay)
        except Exception as e:
            print(f"예상치 못한 에러: {e}")
            raise

async def past_data(sym, tf, limit, since=None):
    async def _past_data():
        binance = get_binance()
        coininfo = await binance.fetch_ohlcv(symbol=sym, 
            timeframe=tf, since=since, limit=limit)

        df = pd.DataFrame(coininfo, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
        df.set_index('datetime', inplace=True)
        await binance.close()
        return df
    return await retry_on_error(_past_data)


# async def past_data(sym, tf, limit, since=None):
#     for attempt in range(3):  # 3번 재시도
#         try:
#             binance = get_binance()
#             coininfo = await binance.fetch_ohlcv(
#                 symbol=sym, 
#                 timeframe=tf, 
#                 since=since, 
#                 limit=limit
#             )
            
#             df = pd.DataFrame(coininfo, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
#             df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
#             df.set_index('datetime', inplace=True)
            
#             await binance.close()
#             return df
            
#         except ccxt.NetworkError as e:
#             print(f"Network error (attempt {attempt + 1}): {e}")
#             if binance:
#                 await binance.close()
#             if attempt < 2:
#                 await asyncio.sleep(2 ** attempt)
#             else:
#                 raise
#         except Exception as e:
#             if binance:
#                 await binance.close()
#             raise
        
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


# def timing_to_close(sym, max_loss, N, t, lev):
#     curr_pnl, profit = asyncio.run(get_curr_pnl(sym.replace("/", "")))
#     if curr_pnl != 0:
#         if curr_pnl < max_loss:# or (t > 25*60 and curr_pnl > 0.1*lev):
#             print(f"\n!!! Close: {curr_pnl}%")
#             return True, curr_pnl, profit

#     return False, curr_pnl, profit


def trailing_stop(curr_pnl, tp, sl, famt=0.3):
    if curr_pnl > tp:
        tp += famt
        sl += famt
        return tp, sl


def get_running_syms():
    with open("running_syms.txt", 'r') as f:
        syms = f.read()
    return eval(syms)

def pop_from_existing_positions(sym):
    syms = get_running_syms()
    syms.pop(syms.index(sym))
    with open('running_syms.txt', 'w') as f:
        f.write(str(syms))

def add_to_existing_positions(sym):
    syms = get_running_syms()
    if sym in syms:
        return True
    syms.append(sym)
    with open('running_syms.txt', 'w') as f:
        f.write(str(syms))


def log_wallet_history(balance):
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
    timestamps = [datetime.fromtimestamp(ts) for ts in wallet_info[0]]
    
    plt.figure(figsize=(8, 4))
    plt.plot(timestamps, wallet_info[1], 'k-', label='Total Wallet Balance')
    plt.plot(timestamps, wallet_info[2], 'b-', label='Total Margin Balance')
    
    ax = plt.gca()
    data_points = len(timestamps)
    if data_points <= 8:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    else:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
        
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
    plt.xticks(rotation=0)
    plt.legend()
    plt.title('Wallet Balance History')
    plt.xlabel('Time')
    plt.ylabel('Balance (USDT)')
    plt.tight_layout()
    plt.savefig("wallet_log.png", dpi=100, bbox_inches='tight')
    plt.close()
    
    
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
        },
        'timeout': 60000,  # 60초 타임아웃
        'rateLimit': 1200,  # 요청 간격 조정
    })
    return binance

# 예쁜 print를 위해
def _b(str):
    return f"{Fore.BLUE}{str}{Style.RESET_ALL}"
def _r(str):
    return f"{Fore.RED}{str}{Style.RESET_ALL}"
def y(str):
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