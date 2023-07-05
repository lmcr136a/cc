import datetime
import numpy as np
import pandas as pd
import mplfinance as mpf
import time
import argparse
import matplotlib.pyplot as plt
import ccxt 
from utils import *

class Trader():
    def __init__(self, symbol=None, symnum=1) -> None:
        self.binance = get_binance()
        self.symnum = symnum
        self.cond1 = 0.2
        self.pre_cond = 0.1
        self.buying_cond = self.cond1
        self.order_num = 30
        self.tf = '1m'
        self.lev = 20
        self.wins = [1, 11, 20, 40]
        self.limit = self.wins[-1]*10           # for past_data
        self.max_loss = max(-2*self.lev, -30)   # 마이너스인거 확인
        self.min_profit = 0.25*self.lev          # 20 일때 5%
        
        if not symbol:
            symbol = select_sym(self.binance, self.buying_cond, self.pre_cond, self.tf, self.limit, self.wins)
        self.sym = symbol

        # 갑자기 올랐을때/ 떨어졌을 때 satisfying_profit 넘으면 close
        self.satisfying_profit = 0.6*self.lev   # 20 일때 12%

        self.time_interval = 2
        self.tf_ = int(self.tf[:-1])
        self.set_lev = True                     # 기존거 가져오는경우 다시 set하면 에러나기때문
        self.binance.load_markets()
        self.last_order = 0
        self.inquire_curr_info(init=True)

        if self.set_lev:  # 기존꺼 가져오는 경우가 아니라 걍 첨에 시작하는 경우
            self.amount = cal_compound_amt(self.wallet_usdt, self.lev, self.price, self.symnum)
        
        print(f"{'*'*50}\nwallet:{round(self.wallet_usdt, 3)}  tf: {self.tf}  lev:{self.lev}  \
\namt: {self.amount}  inf: {self.wins}\n{'*'*50}  [[{self.max_loss}~{self.min_profit}]]")

        market_status = bull_or_bear(past_data(self.binance,sym=self.sym, tf='2h', limit=50))
        print(f"{self.sym} {market_status} MARKET")
        if market_status == "BULL":     # 상승장이면
            self.buying_cond = -0.3      # 원래 -buying_cond 보다 낮아야 살 수 있던걸 바꿔줌
            self.pre_cond = -0.75
        elif market_status == "BEAR":   # 하락장이면
            self.buying_cond = -0.3     # 원래 buying_cond 보다 높아야 살 수 있던걸 바꿔줌
            self.pre_cond = -0.75

    def update_wallet(self, balance=None):
        if not balance:
            balance = self.binance.fetch_balance()
        # USDT
        for asset in balance['info']['assets']:
            if asset['asset'] == 'USDT':
                self.wallet_usdt = float(asset['availableBalance'])

    ## 현재 정보 조회
    def inquire_curr_info(self, init=False):
        balance = self.binance.fetch_balance()
        positions = balance['info']['positions']
        
        log_wallet_history(balance)
        # USDT
        self.update_wallet(balance)

        for position in positions:
            if position["symbol"] == self.sym.replace("/", ""):
                amt = float(position['positionAmt'])
                if amt == 0:
                    self.status = None
                    return 0
                pnl, profit = self.get_curr_pnl(self.sym.replace("/", ""))
                if init:
                    self.init_ckpt(position, pnl)
                    self.set_lev = False


    def init_ckpt(self, position, pnl):
        amt = float(position['positionAmt'])
        self.lev = float(position['leverage'])
        self.amount = amt
        print(f"{self.sym} Current PNL:", pnl, "% Leverage: ", self.lev, ' Amt: ', amt)
        if amt < 0:
            self.status = SHORT
        elif amt > 0:
            self.status = LONG

    def inquire_curr_price(self):
        info = self.binance.fetch_ticker(self.sym)
        return info['average']

    def e_long(self, close=False):  # 오를것이다
        self.binance.load_markets()
        price = self.inquire_curr_price()
        market = self.binance.markets[self.sym]
        if self.set_lev:
            resp = self.binance.set_leverage(
                symbol=market['id'],
                leverage=self.lev,
            )
        order = self.binance.create_market_buy_order(
            symbol=self.sym,
            amount=np.abs(self.amount),
            )
        self.order_id = order['id']
        
    def e_short(self, close=False):  # 내릴것이다
        self.binance.load_markets()
        price = self.inquire_curr_price()
        market = self.binance.markets[self.sym]
        if self.set_lev:
            resp = self.binance.set_leverage(
                symbol=market['id'],
                leverage=self.lev,
            )
        order = self.binance.create_market_sell_order(
            symbol=self.sym,
            amount=np.abs(self.amount),
            )
        self.order_id = order['id']


    def run(self):
        print("\nStarting status: ", self.status)
        transactions = []
        tr = {}
        tr['ent'] = [0, 0]
        iter = 0
        self.anxious = 1
        while len(transactions) < self.order_num:
            m1, m2, m3, m4 = get_ms(self.binance, self.sym, self.tf, self.limit, self.wins)

            if not self.status:
            
                self.status = timing_to_position(self.sym, buying_cond=self.buying_cond, pre_cond=self.pre_cond)

                if self.status == LONG:
                    self.e_long()
                elif self.status == SHORT:
                    self.e_short()
                
                if self.status:
                    tr['ent'] = [iter, m1[-1]]

            else :
                howmuchtime = iter - tr['ent'][0]
                # 시간이 오래 지날수록 욕심을 버리기
                if (howmuchtime)%(2*3600/self.time_interval) == 0: # 1시간
                    self.anxious *= 0.8
                    self.anxious = max(self.anxious, 1.2/self.satisfying_profit)
                satisfying_price = self.satisfying_profit*self.anxious
                curr_cond = get_curr_cond(m1, period=500)
                does_m4_turnning = m4_turn(self.status, m2, m3, m4)
                
                # curr pnl을 return하는건 그냥임
                curr_pnl = timing_to_close(binance=self.binance, sym=self.sym, status=self.status, 
                        curr_cond=curr_cond, does_m4_turnning=does_m4_turnning, m1=m1, satisfying_price=satisfying_price, 
                        max_loss=self.max_loss, min_profit=self.min_profit, cond1=self.cond1, howmuchtime=howmuchtime)
                
                if curr_pnl:
                    if self.status == LONG:
                        self.e_short(close=True)
                    else:
                        self.e_long(close=True)
                        
                    tr['close'] = [iter, m1[-1]]
                    tr = close(self.status, tr, curr_pnl)
                    transactions.append(tr)
                    tr = {}
                    self.status = None
                    self.anxious = 1
                    with open("transactions.txt", 'w') as f:
                        f.write(str(transactions))
                        
                    self.update_wallet()
                    price = float(self.inquire_curr_price())
                    self.amount = cal_compound_amt(self.wallet_usdt, self.lev, price, self.symnum)

            time.sleep(self.time_interval)
            iter += 1



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol',
                        '-s',
                        default=None,
                        type=str,
                        )
    parser.add_argument('--symnum',
                        '-n',
                        default=None,
                        type=str,
                        )
    args = parser.parse_args()

    trader = Trader(args.symbol, args.symnum)
    trader.run()