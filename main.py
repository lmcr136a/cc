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
        self.sym = 'BNB/USDT'
        self.order_num = 30
        self.tf = '1m'
        self.tf_ = int(self.tf[:-1])
        self.lev = 5
        self.set_lev = True  # 기존거 가져오는경우 다시 set하면 에러나기때문
        self.amount = 0.01
        self.wins = [1, 11, 20, 40]
        self.limit = self.wins[-1]*10    # for past_data
        self.max_loss = -20             # 마이너스인거 확인
        self.min_profit = 5
        print(f"{'*'*50}\ntf: {self.tf}  lev:{self.lev}  amt: {self.amount}  inf: {self.wins}\n{'*'*50}  [[{self.max_loss}~{self.min_profit}]]")
        self.binance.load_markets()
        self.last_order = 0
        self.inquire_curr_info(init=True)
        

    ## 현재 정보 조회
    def inquire_curr_info(self, init=False):
        balance = self.binance.fetch_balance()
        positions = balance['info']['positions']

        for position in positions:
        #     print(position)
            if position["symbol"] == self.sym.replace("/", ""):
                amt = float(position['positionAmt'])
                if amt == 0:
                    self.status = None
                    return 0
                pnl = self.get_curr_pnl(self.sym.replace("/", ""))
                if init:
                    self.init_ckpt(position, pnl)
                    self.set_lev = False


    def init_ckpt(self, position, pnl):
        amt = float(position['positionAmt'])
        self.lev = float(position['leverage'])
        self.amount = amt
        print("Current PNL:", pnl, "% Leverage: ", self.lev, ' Amt: ', amt)
        if amt < 0:
            self.status = 'Short'
        elif amt > 0:
            self.status = 'Long'
        else:
            print("뭔가이상해..")
            self.status = None

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



    def get_curr_pnl(self, sym):
        wallet = self.binance.fetch_balance(params={"type": "future"})
        positions = wallet['info']['positions']
        for pos in positions:
            if pos['symbol'] == sym:
                pnl = float(pos['unrealizedProfit'])/abs(float(pos['positionAmt']))/float(pos['entryPrice'])*100*float(pos['leverage'])
                return pnl



    def e_long(self):  # 오를것이다
        self.binance.load_markets()
        price = self.inquire_curr_price(self.sym)
        # price = price*(1-a)
        market = self.binance.markets[self.sym]
        if self.set_lev:
            resp = self.binance.set_leverage(
                symbol=market['id'],
                leverage=self.lev,
            )
        params = {'timeInForce': 'IOC',}
        order = self.binance.create_market_buy_order(
            symbol=self.sym,
            amount=np.abs(self.amount),
            # price=price,
            # params=params,
            )
        self.order_id = order['id']
        if not order["postOnly"]:
            print("\n\n\n!It's not post-only order!")
        print(f'Posted [price: {price}]')
        
    def e_short(self):  # 내릴것이다
        self.binance.load_markets()
        price = self.inquire_curr_price(self.sym)
        market = self.binance.markets[self.sym]
        if self.set_lev:
            resp = self.binance.set_leverage(
                symbol=market['id'],
                leverage=self.lev,
            )

        params = {'timeInForce': 'IOC',}
        order = self.binance.create_market_sell_order(
            symbol=self.sym,
            amount=np.abs(self.amount),
            # price=price,
            # params=params,
            )
        self.order_id = order['id']
        if not order["postOnly"]:
            print("\n\n\n!It's not post-only order!")
        print(f'Posted [price: {price}]')


    def get_ms(self):
        df = self.past_data(sym=self.sym, tf=self.tf, limit=self.limit)
        m1 = df['close'].rolling(window=self.wins[0]).mean()
        m2 =  df['close'].rolling(window=self.wins[1]).mean()
        m3 = df['close'].rolling(window=self.wins[2]).mean()
        m4 = df['close'].rolling(window=self.wins[3]).mean()
        return m1, m2, m3, m4

    def run0612(self):
        status = self.status
        print("\nStarting status: ", status)
        transactions = []
        tr = {}
        tr['ent'] = [0, 0]
        order_i = 0
        iter = 0
        cond1, cond2 = 0.2, 0.1
        while order_i < self.order_num:
            i = -1
            m1, m2, m3, m4 = self.get_ms()

            if not status:
            
                turnning_shape = whether_turnning(m2, m3, m4, ref=0.001*0.01, ref2=0.01*0.01)  # u or n or None
                # val = self.get_curr_cond(m1)
                val = self.get_curr_conds()
                pre_cond = np.mean(val[1:])
                
                if turnning_shape == 'u' and val[0] < -cond1 and pre_cond < -cond2:
                    status = "Long"
                    self.e_long()
                elif turnning_shape == 'n' and val[0] > cond1 and pre_cond > cond2:
                    status = "Short"
                    self.e_short()
                
                if status:
                    tr['ent'] = [i, m1[i]]

            else :
                curr_pnl = self.get_curr_pnl(self.sym.replace("/", ""))
                curr_cond = get_curr_cond(m1)
                if iter % 2000 == 0:
                    print("curr_pnl: ", round(curr_pnl, 5), "%")
                iter += 1
                if curr_pnl < self.max_loss \
                    or\
                (
                    curr_pnl > self.min_profit \
                        and\
                    timing2_close(status, m2, m3, m4)
                        and\
                    (
                        (curr_cond < cond1 and status == "Short") \
                            or\
                        (curr_cond > cond1 and status == "Long")
                    )
                ):
                    print(curr_pnl, status)
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
            time.sleep(2)

    def get_curr_conds(self, tfs= ['1m', '3m', '5m', '15m']):
        pos_val = []
        for tf in tfs:
            # tf = ex) '1m'
            df = self.past_data(self.sym, tf, limit=1500)
            m = df['close'].rolling(window=1).mean()
            v = get_curr_cond(m)
            pos_val.append(v)
        return pos_val

def minmax(m):
    m = (m - m.min())/(m.max() - m.min())
    return m

def whether_turnning(m2, m3, m4, ref=0.001*0.01, ref2=0.02*0.01):
    i = -1
    curr_m2on3 = (m2[i] - m3[i])/m2[i] > ref
    curr_m4 = np.abs((m4[i] - m4[i-2])/m4[i])
    pre_m4 = np.abs((m4[i-2] - m4[i-4])/m4[i-2])
    p_m2on3 = []
    p_m2under3 = []
    m4incs = []
    m4decs = []
    for pre in range(1, 10):
        m2on3 = (m2[i-pre] - m3[i-pre])/m2[i-pre] > ref
        m2under3 = (m2[i-pre] - m3[i-pre])/m2[i-pre] < -ref
        m4inc = (m4[i] - m4[i-pre])/m4[i] > 0
        m4dec = (m4[i] - m4[i-pre])/m4[i] < 0
        p_m2on3.append(m2on3)
        p_m2under3.append(m2under3)
        m4incs.append(m4inc)
        m4decs.append(m4dec)

    p_m2on3 = np.all(p_m2on3)
    p_m2under3 = np.all(p_m2under3)
    m4_increasing = np.all(m4incs)
    m4_decreasing = np.all(m4decs)
    
    m23_start_inc = p_m2under3 and curr_m2on3 
    m23_start_dec = p_m2on3 and not curr_m2on3
            
    if curr_m4 <pre_m4 and curr_m4 < ref2:
        if m23_start_inc and m4_decreasing:
            return 'u'
        elif m23_start_dec and m4_increasing:
            return 'n'
    return None

def get_curr_cond(m):
    mm = minmax(m)
    m_mean = np.mean(mm)
    return mm[-1] - m_mean
    
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

def timing2_close(status, m2, m3, m4, ref=0):
    i = -1
    m4_inc1 = m4[i-2] - m4[i-3] 
    m4_inc2 = m4[i] - m4[i-1] 
    if status == "Long": # n
        m4_turn = m4_inc1>ref and m4_inc2 <ref
    else:
        m4_turn = m4_inc1<ref and m4_inc2 >ref
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
    trader.run0612()