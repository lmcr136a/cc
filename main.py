import datetime
import numpy as np
import pandas as pd
import mplfinance as mpf
import time
import argparse
import matplotlib.pyplot as plt
import ccxt 
from utils import *

"""
TODO: 
1. 15m 따라서 숏/롱 포지션 바꾸기
2. 에러처리 try except를 while처리하기 
3. wallet log 잘 찍히도록
"""

class Trader():
    def __init__(self, symbol=None, symnum=1) -> None:
        self.binance = get_binance()
        self.symnum = float(symnum)
        self.other_running_sym_num = 0
        self.cond1 = 0.4
        self.pre_cond = 0.0
        self.buying_cond = self.cond1
        self.order_num = 1                      # 거래 한번만
        self.tf = '3m'
        self.lev = 20
        self.wins = [1, 7, 15, 20]              # 3번째
        self.limit = self.wins[-1]*10           # for past_data
        self.max_loss = max(-2*self.lev, -25)   # 마이너스인거 확인
        self.min_profit = 0.25*self.lev          # 20 일때 5%
        
        if not symbol:
            symbol = select_sym(self.binance, self.buying_cond, self.pre_cond, 
                                self.tf, self.limit, self.wins, self.symnum)
        self.sym = symbol

        # 갑자기 올랐을때/ 떨어졌을 때 satisfying_profit 넘으면 close
        self.satisfying_profit = 0.5*self.lev   # 20 일때 16%

        self.time_interval = 2
        self.tf_ = int(self.tf[:-1])
        self.set_lev = True                     # 기존거 가져오는경우 다시 set하면 에러나기때문
        self.binance.load_markets()
        self.last_order = 0
        self.inquire_curr_info(init=True)

        if self.set_lev:  # 기존꺼 가져오는 경우가 아니라 걍 첨에 시작하는 경우
            self.amount = cal_compound_amt(self.wallet_usdt, self.lev, float(self.inquire_curr_price()), self.symnum)
        
        print(f"{'*'*50}\nwallet:{round(self.wallet_usdt, 3)}  tf: {self.tf}  lev:{self.lev}  \
\namt: {self.amount}  inf: {self.wins}\n{'*'*50}  [[{self.max_loss}~{self.min_profit}]]")

        actions = inspect_market(self.binance, self.sym, self.satisfying_profit, self.buying_cond)
        self.short_only, self.long_only, self.buying_cond, self.satisfying_profit = actions

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
            amt = float(position['positionAmt'])
            if amt > 0 and position["symbol"] != self.sym.replace("/", ""):
                self.other_running_sym_num += 1
        self.symnum -= self.other_running_sym_num 
        print(f"other_running_sym_num: {self.other_running_sym_num}  self.symnum: {self.symnum}")
        for position in positions:
            amt = float(position['positionAmt'])
            if position["symbol"] == self.sym.replace("/", ""):
                amt = float(position['positionAmt'])
                if amt == 0:
                    self.status = None
                    return 0
                pnl, profit = get_curr_pnl(self.binance, self.sym.replace("/", ""))
                if init:
                    self.init_ckpt(position, pnl)
                    self.set_lev = False
                    return None


    def init_ckpt(self, position, pnl):
        amt = float(position['positionAmt'])
        self.lev = float(position['leverage'])
        self.amount = amt
        if amt < 0:
            self.status = SHORT
        elif amt > 0:
            self.status = LONG
        print(f"{self.sym} {self.status} Current PNL:", pnl, "% Leverage: ", self.lev, ' Amt: ', amt)
        

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
        iter = 0
        self.anxious = 1
        self.pre_pnls = []
        self.missed_timing = 0
        while 1: 
            m1, m2, m3, m4 = get_ms(self.binance, self.sym, self.tf, self.limit, self.wins)
            
            if self.missed_timing > 10:
                return 0
            if not self.status:
            
                self.status = timing_to_position(self.binance, self.sym, buying_cond=self.buying_cond, pre_cond=self.pre_cond, tf=self.tf, limit=self.limit, wins=self.wins)

                try:
                    if self.status == LONG:
                        self.e_long()
                    elif self.status == SHORT:
                        self.e_short()
                except Exception as error:
                    print(error)
                    self.status = None

                if not self.status:
                    self.missed_timing += 1

            else :
                # 시간이 오래 지날수록 욕심을 버리기
                if (iter)%((3600/2)/self.time_interval) == 0 and iter > 0: # 3600 == 1h
                    loss_count = np.sum(np.where(np.array(self.pre_pnls) < 0, 1, 0))
                    loss_ratio = loss_count/len(self.pre_pnls)  # 값이 크면 계속 잃었던 것
                    self.satisfying_profit *= 0.8
                    self.satisfying_profit = round(max(self.satisfying_profit, self.min_profit), 2)

                m4_shape = m4_turn(m4)
                
                # curr pnl을 return하는건 그냥임
                close_position, curr_pnl = timing_to_close(binance=self.binance, sym=self.sym, status=self.status, 
                        m4_shape=m4_shape, m1=m1, satisfying_price=self.satisfying_profit, 
                        max_loss=self.max_loss, min_profit=self.min_profit, cond1=self.cond1, howmuchtime=iter)
                self.pre_pnls.append(curr_pnl)
                
                # 포지션과 반대되는 방향으로 m3그래프가 변하면
                # 현재 포지션 정리, 반대 포지션으로 바꿈
                # if (iter)%((3600/4)/self.time_interval) == 0 and iter > 0: # 3600 == 1h, every 15min
                have2chg = isit_wrong_position(m2, self.status)
                
                if close_position:
                    try:
                        if self.status == LONG:
                            self.e_short(close=True)
                        else:
                            self.e_long(close=True)
                        return 0  # finish the iteration
                    except Exception as error:
                        print(error)
                
                if have2chg:
                    complete = False
                    print(f"I think {self.status} position is wrong.. I'll change it to opposite pos.")
                    while not complete:
                        try:
                            if self.status == LONG:
                                self.e_short(close=True)
                                self.e_short()
                                self.status = SHORT
                                print("Changed to SHORT")
                            else:
                                self.e_long(close=True)
                                self.e_long()
                                self.status = LONG
                                print("Changed to LONG")
                            complete = True
                            time.sleep(1)
                        except Exception as error:
                            print(error)


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
                        default=1,
                        type=int,
                        )
    parser.add_argument('--re_execution',
                        '-r',
                        default=True,
                        type=bool,
                        )
    args = parser.parse_args()
    sym = args.symbol if not args.symbol or '/USDT' in args.symbol else args.symbol + '/USDT'
    while 1:
        trader = Trader(sym, args.symnum)
        trader.run()
        sym = None
        if not args.re_execution:
            break