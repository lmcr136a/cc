import importlib
import numpy as np
import time
import utils
from utils import *

"""
1.2
"""
class Trader():
    def __init__(self, symbol=None, symnum=1) -> None:
        self.binance = get_binance()

        self.symnum = float(symnum)
        self.other_running_sym_num = 0
        self.lev = 10  # 0.07*lev = 0.7% 가 수수료
        self.stoploss = -0.6*self.lev                     # 마이너스인거 확인
        self.takeprofit = 0.9*self.lev   # 3 * 10
        self.limit_amt_ratio = 0
        
        if not symbol:
            symbol, self.position_to_by = select_sym(self.binance, self.symnum)
        self.sym = symbol

        self.time_interval = 3
        self.set_lev = True                     # 기존거 가져오는경우 다시 set하면 에러나기때문
        self.binance.fetch_markets()
        self.inquire_curr_info(init=True)

        if self.set_lev:  # 기존꺼 가져오는 경우가 아니라 걍 첨에 시작하는 경우
            self.amount = cal_compound_amt(self.wallet_usdt, self.lev, float(self.inquire_curr_price()), self.symnum)
        
        self.filled = False
        print(f"{'*'*50}\nwallet:{round(self.wallet_usdt, 3)}  lev:{self.lev} [[{self.stoploss}~{self.takeprofit}]]")


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
            if abs(amt) > 0 and position["symbol"] != self.sym.replace("/", ""):
                self.other_running_sym_num += 1
        self.symnum -= self.other_running_sym_num 
        print(f"other_running_sym_num: {self.other_running_sym_num}  self.symnum: {self.symnum}")

        if init:
            amt = self.get_curr_sym_amt()
            print(amt, self.sym)
            if abs(amt) > 0: 
                pnl, profit = get_curr_pnl(self.binance, self.sym.replace("/", ""))
                self.init_ckpt(amt, pnl)
                self.set_lev = False
                return 0
        self.status = None

    def get_curr_sym_amt(self,):
        balance = self.binance.fetch_balance()
        positions = balance['info']['positions']
        for position in positions:
            if position["symbol"] == self.sym.replace("/", ""):
                amt = float(position['positionAmt'])
                return amt
        return 0
        

    def init_ckpt(self, amt, pnl):
        self.amount = amt
        if amt < 0:
            self.status = SHORT
        elif amt > 0:
            self.status = LONG
        print(f"{self.sym} {self.status} Current PNL:", pnl, "% Leverage: ", self.lev, ' Amt: ', amt)
        

    def inquire_curr_price(self):
        info = self.binance.fetch_ticker(self.sym)
        return info['close']

    def prep_order(self,):
        self.binance.load_markets()
        raw_price = self.inquire_curr_price()
        point_len = len(str(raw_price).split(".")[-1])
        price = round(raw_price*(1-self.limit_amt_ratio), point_len)  

        print(f"Try to [{self.sym}] price:{raw_price}->{price} LONG order amt:{self.amount}")

        market = self.binance.markets[self.sym]
        if self.set_lev:
            resp = self.binance.set_leverage(
                symbol=market['id'],
                leverage=self.lev,
            )
        return price

    def open_order(self):
        if self.status == LONG:
            side = "buy"
        elif self.status == SHORT:
            side = "sell"
        price = self.prep_order()
        order = self.binance.create_order(
            symbol=self.sym, type="limit", side=side, amount=np.abs(self.amount), price=price)
        self.order_id = order['id']
        
    def close_order(self):
        if self.status == LONG:
            side = "sell"
        elif self.status == SHORT:
            side = "buy"
        order = self.binance.create_order(
            symbol=self.sym, type="market", side=side, amount=np.abs(self.amount))
        self.order_id = order['id']

    def run(self):
        iter = 0
        self.anxious = 1
        self.pre_pnls = []
        self.missed_timing = 0
        while 1:
            
            if self.missed_timing > 3:
                return 0
            if not self.status:
            
                self.status = self.position_to_by

                try:
                    self.open_order()
                    
                except Exception as error:
                    print(error)
                    self.status = None
                    if "Leverage" in str(error):
                        self.lev = round(0.5*self.lev)

                if not self.status:
                    self.missed_timing += 1

            else :
                curr_amt = self.get_curr_sym_amt()
                if len(self.pre_pnls) > 10 and not curr_amt:  # limit order 안사짐
                    self.binance.cancel_order(id=int(self.order_id), symbol=self.sym)
                    return self.sym 
                
                
                close_position, curr_pnl = timing_to_close(binance=self.binance, sym=self.sym, 
                                                           satisfying_profit=self.takeprofit, 
                                                           max_loss=self.stoploss)
                self.pre_pnls.append(curr_pnl)
                if curr_pnl == -100:
                    return self.sym
                
                if close_position:
                    try:
                        self.close_order()
                        return self.sym  # finish the iteration
                    except Exception as error:
                        print(error)
                
                

            time.sleep(self.time_interval)
            iter += 1
