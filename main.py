import datetime
import numpy as np
import pandas as pd
import mplfinance as mpf
import time
import matplotlib.pyplot as plt
import ccxt 
from visualization import *

"""
본전가격: lev* 0.05%            ex) lev2: 0.1%  lev4: 0.2%
한번의 거래에서 lev*0.1% 먹기 전까지는 버텨보기
만약 lev*0.3% 이상 손해면 극대점에서 팔기
상승하는거 살때: [2, 3, 6]이면 23이 교차하는데 6이 상승중일때 사기
상승하는거 팔때: [2, 3, 6]으로 해서 6이 상승중이면 23 교차해도 안팔기 (곧 또 상승할거라 예측)
            대신 6이 하락하면 팔기

1. 함수 정리
2. 중간에 껐다 켜도 현상태 받아와서 똑같이 지속되도록
3. visualization
"""
class Trader():
    def __init__(self) -> None:
        with open("a.txt") as f:
            lines = f.readlines()
            api_key = lines[0].strip()
            secret  = lines[1].strip()

        self.binance = ccxt.binance(config={
            'apiKey': api_key, 
            'secret': secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future'
            }
        })
        self.sym = 'BTC/USDT'
        self.order_num = 30
        self.tf = '1m'
        self.tf_ = int(self.tf[:-1])
        self.n = 5       # 걍 3보다 크면 됨
        self.lev = 2
        self.amount = 0.001
        self.wins = [1, 2, 11, 17]
        self.limit = self.wins[-1]+5  # for past_data
        print(f"{'*'*50}\ntf: {self.tf}  lev:{self.lev}  amt: {self.amount}  inf: {self.wins}\n{'*'*50}")
        self.binance.load_markets()
        self.inquire_my_wallet()
        self.last_order = 0

    ## 현재 정보 조회
    def inquire_curr_info(self, sym="BTC/USDT"):
        info = self.binance.fetch_ticker(sym)
        print(info)
        
    def inquire_curr_price(self, sym="BTC/USDT"):
        info = self.binance.fetch_ticker(sym)
        return info['average']
        
    def past_data(self, sym, tf, limit, since=None):
        coininfo = self.binance.fetch_ohlcv(
            symbol=sym, 
            timeframe=tf, 
            since=since, 
            limit=limit
        )
        df = pd.DataFrame(coininfo, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
        df.set_index('datetime', inplace=True)
        return df

    def inquire_my_wallet(self, justshow=None):
        wallet = self.binance.fetch_balance(params={"type": "future"})
        keys = ['used', 'total']
        for k in keys:
            if justshow:
                s = justshow.split("/")[0]
                print(f"{k} USDT: {wallet[k]['USDT']}  {s}: {wallet[k][s]}")
            else:
                print(f"\n{k}:\n {wallet[k]}")

    def e_long(self):  # 오를것이다
        self.binance.load_markets()
        a = 0.01*0.012342342423423
        price = self.inquire_curr_price(self.sym)
        # price = price*(1-a)
        market = self.binance.markets[self.sym]

        resp = self.binance.set_leverage(
            symbol=market['id'],
            leverage=self.lev,
        )
        params = {'timeInForce': 'GTC',}
        order = self.binance.create_limit_buy_order(
            symbol=self.sym,
            amount=self.amount,
            price=price,
            params=params,
            )
        self.order_id = order['id']
        if not order["postOnly"]:
            print("\n\n\n!It's not post-only order!")
        print(f'Posted [price: {price}]')
        
    def e_short(self):  # 내릴것이다
        self.binance.load_markets()
        price = self.inquire_curr_price(self.sym)
        market = self.binance.markets[self.sym]

        resp = self.binance.set_leverage(
            symbol=market['id'],
            leverage=self.lev,
        )
        
        params = {'timeInForce': 'GTC',}
        order = self.binance.create_limit_sell_order(
            symbol=self.sym,
            amount=self.amount,
            price=price,
            params=params,
            )
        self.order_id = order['id']
        if not order["postOnly"]:
            print("\n\n\n!It's not post-only order!")
        print(f'Posted [price: {price}]')

    def is_remaining_order(self):        
        balance = self.binance.fetch_balance()
        positions = balance['info']['positions']
        print(positions)
        for position in positions:
            if position["symbol"] == "BTCUSDT":
                print(position)
                return True
        return False


    def whether_increasing(self):
        d1m = self.past_data(sym=self.sym, tf=self.tf, limit=self.limit)
        ma1 = d1m['close'].rolling(window=self.wins[0]).mean()[-1]
        ma2 =  d1m['close'].rolling(window=self.wins[1]).mean()[-1]
        show_default_graph(d1m, d1m['close'].rolling(window=self.wins[0]).mean(), d1m['close'].rolling(window=self.wins[1]).mean(),
                           n=10)
        return ma1 - ma2 > 0

        
    def run(self):
        previous = self.whether_increasing()
        print(f'Start With {previous} increasing status')
        status = 'Start'
        order_i = 0
        iter = 0
        self.e_long()
        status = 'Long'
        while order_i < self.order_num:
            increasing1 = self.whether_increasing()
            if iter%10 == 0:
                print(f"{iter}) Increasing? {increasing1}, Increased previous? {previous}")

            if not previous and increasing1: # 상승하기시작
                time.sleep(self.tf_*30)
                increasing1 = self.whether_increasing()
                if not previous and increasing1:

                    if status == 'Start':
                        self.e_long()
                        print("Long stance")
                    elif status == 'Short':
                        self.e_long()
                        self.e_long()
                        print("Short -> Long stance")
                    else:
                        print('Error....', status)
                    status = 'Long'
                    order_i+= 1
                    self.last_order = iter
            elif previous and not increasing1: # 떨어지기 시작
                time.sleep(self.tf_*30)
                increasing1 = self.whether_increasing()
                if previous and not increasing1:
                    
                    if status == 'Start':
                        self.e_short()
                        print("Short stance")
                    elif status == 'Long':
                        self.e_short()
                        self.e_short()
                        print("Long -> Short stance")
                    else:
                        print('Error....', status)
                    status = 'Short'
                    order_i += 1
                    self.last_order = iter
            previous = increasing1

            if order_i % 10 == 0 and iter%100 == 0:
                print(order_i, "-th order")
                self.inquire_my_wallet(justshow=self.sym)
            time.sleep(2)
            iter += 1
            
        if status == "Short":
            self.e_long()
        elif status == "Long":
            self.e_short()
        else:
            print("Error!!!", status)
            
    def get_ms(self):
        df = self.past_data(sym=self.sym, tf=self.tf, limit=self.limit)
        m1 = df['close'].rolling(window=self.wins[0]).mean()[-self.n:]
        m2 =  df['close'].rolling(window=self.wins[1]).mean()[-self.n:]
        m3 = df['close'].rolling(window=self.wins[2]).mean()[-self.n:]
        m4 = df['close'].rolling(window=self.wins[3]).mean()[-self.n:]
        return m1, m2, m3, m4

    def run0531(self):
        status = None
        transactions = []
        tr = {}
        order_i = 0
        iter = 0
        while order_i < self.order_num:
            i = -1
            m1, m2, m3, m4 = self.get_ms()
            if not status:
                m2on3 = m2[i] - m3[i] > 0    
                p_m2on3 = m2[i-1] - m3[i-1] > 0
                m4_increasing = m4[i] - m4[i-1] > 0
                m23_start_inc = not p_m2on3 and m2on3
                m23_start_dec = p_m2on3 and not m2on3
                
                if m23_start_inc and m4_increasing:
                    status = "Long"
                    self.e_long()
                elif m23_start_dec and not m4_increasing:
                    status = "Short"
                    self.e_short()
                
                if status:
                    tr['ent'] = [i, m1[i]]
                    ent_price = m1[i]                

                    positionamt = 0
                    start = time.time()
                    waiting = 0
                    while positionamt == 0 and waiting < self.tf_*60:
                        waiting = time.time() - start
                        balance = self.binance.fetch_balance()
                        positions = balance['info']['positions']

                        for position in positions:
                            if position["symbol"] == self.sym.replace("/", ""):
                                positionamt = position['positionAmt']
                    if positionamt == 0:
                        resp = self.binance.cancel_order(
                                id=self.order_id,
                                symbol=self.sym
                            )
                    self.inquire_my_wallet(justshow=self.sym)

            else :
                p = 2*(-0.5+int(status=="Long"))
                curr_pnl = p*(m1[i] - ent_price)/ent_price*100
                if iter % 200 == 0:
                    print("curr_pnl: ", curr_pnl)
                if curr_pnl < -0.5 or\
                (
                    curr_pnl > 0.2 and\
                    timing2_close(status, m2, m3, m4, i)
                ):
    #                 print(curr_pnl, status)
                    if status == 'Long':
                        self.e_short()
                    else:
                        self.e_long()
                        
                    tr['close'] = [i, m1[i]]
                    tr = close(status, tr)
                    transactions.append(tr)
                    tr = {}
                    status = None
        with open("transactions.txt", 'w') as f:
            f.write(transactions)

def show_total_pnl(transactions):
    pnls=[]
    for tr in transactions:
        pnls.append(float(tr['pnl'][:-1]))
    total_pnl = np.sum(pnls)
#     print(f"{total_pnl} %")
    return total_pnl
""" -----Rule-----
1. m2 ^ & m3 v & m4 /: 롱 진입
   m4 하락하기 시작: 롱 정리
2. m2 v & m3 ^ & m4 \: 숏 진입 
   m4 상승하기 시작: 숏 정리
3. 0.05% 이상 수익이 아니면 안팔기
4. 0.1% 이상 손실이면 팔기
"""

def timing2_close(status, m2, m3, m4, i):
    if i < 2:
        return False
    m4_inc1 = m4[i-1] - m4[i-2] 
    m4_inc2 = m4[i] - m4[i-1] 
    m4_turn = m4_inc1 * m4_inc2 < 0
    return m4_turn

def close(pos, tr, lev=1):
    p=1
    if pos == "Short":
        p = -1
    tr['position'] = pos
    profit = p*(tr['close'][1] - tr['ent'][1])/tr['ent'][1] # 1 means price
    profit = 100*profit - 0.08
    tr['pnl'] = str(round(lev*profit, 4))+"%"
    return tr
if __name__ == "__main__":
    trader = Trader()
    trader.run0531()