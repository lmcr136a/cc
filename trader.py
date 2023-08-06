import importlib
import numpy as np
import time
import utils
from utils import *

importlib.reload(utils)  #> TODO: 에러많이남   

class Trader():
    def __init__(self, symbol=None, symnum=1) -> None:
        self.binance = get_binance()
        self.symnum = float(symnum)
        self.other_running_sym_num = 0
        self.tf = '3m'
        self.tf_ = int(self.tf[:-1])
        self.lev = 20
        self.wins = [1, 8, 15, 70]              # 3번째
        self.limit = self.wins[-1]*10           # for past_data

        #########################################
        self.anx_pnl = -40
        self.p = 8 *0.01/self.lev
        #########################################

        if not symbol:
            symbol = select_sym(self.binance, self.tf, self.limit, self.wins, self.symnum)

        self.sym = symbol

        self.time_interval = 3
        self.set_lev = True                     # 기존거 가져오는경우 다시 set하면 에러나기때문
        self.binance.load_markets()
        self.inquire_curr_info(init=True)

        if self.set_lev:  # 기존꺼 가져오는 경우가 아니라 걍 첨에 시작하는 경우
            self.amount = cal_compound_amt(self.wallet_usdt, self.lev, float(self.inquire_curr_price()), self.symnum)
        
        print(f"{'*'*50}\nwallet:{round(self.wallet_usdt, 3)}  tf: {self.tf}  lev:{self.lev} amt: {self.amount}")


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
        for position in positions:
            amt = float(position['positionAmt'])
            if position["symbol"] == self.sym.replace("/", ""):
                amt = float(position['positionAmt'])
                if amt == 0:
                    self.status = None
                    return 0
                pnl, profit= get_curr_pnl(self.binance, self.sym.replace("/", ""))
                if init:
                    open_orders = self.binance.fetch_open_orders(
                        symbol=self.sym
                    )
                    for open_order in open_orders:
                        self.order_id = open_order['id']
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
        return info['last']

    def e_long_market(self, close=False):  # 오를것이다
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
        return price
        
    def e_short_market(self, close=False):  # 내릴것이다
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
        return price

    def close_short_limit(self, price, close=False):  # 숏산거 팔때
        self.binance.load_markets()
        price = price * (1 - self.p)
        order = self.binance.create_limit_buy_order(
            symbol=self.sym,
            amount=np.abs(self.amount),
            price= price,
            )
        self.order_id = order['id']

        
    def close_long_limit(self, price, close=False):  # 롱산거 팔때
        self.binance.load_markets()
        price = price * (1 + self.p)
        order = self.binance.create_limit_sell_order(
            symbol=self.sym,
            amount=np.abs(self.amount),
            price= price,
            )
        self.order_id = order['id']


    def cancle_order(self,):
        order = self.binance.cancel_order(
            symbol=self.sym,
            id=self.order_id
        )

    def order_filled(self):
        balance = self.binance.fetch_balance()
        positions = balance['info']['positions']

        for position in positions:
            if position['symbol'] == self.sym.replace("/", ""):
                amt = float(position['positionAmt'])
                # print(amt)
                if abs(amt) == 0:
                    return True
                else:
                    return False
        return False

    def run(self):
        iter = 0
        self.anxious = 1
        pre_pnls, pre_prices = [], []
        self.missed_timing = 0
        pnl_lastupdate = 0
        while 1:
            ms = get_ms(self.binance, self.sym, self.tf, self.limit, self.wins)
            
            if self.missed_timing > 3:
                return 0
            
            if not self.status:
            
                self.status = timing_to_position(self.binance, ms, self.sym, self.tf, pr=True)
                print(self.status)

                try:
                    if self.status == LONG:
                        price = self.e_long_market()
                        time.sleep(0.1)
                        # add_to_existing_positions(LONG)
                        close_func = self.close_long_limit
                        # pop_from_existing_positions(LONG)

                    elif self.status == SHORT:
                        price =self.e_short_market()
                        time.sleep(0.1)
                        # add_to_existing_positions(SHORT)
                        close_func = self.close_short_limit
                        # pop_from_existing_positions(SHORT)
                    close_func(price=price, close=True)
            
                except Exception as error:
                    print(error)
                    self.status = None
                    if "Leverage" in str(error):
                        self.lev = round(0.5*self.lev)
                    else:
                        exit()
                add_to_existing_positions(self.status)
            else:
                # checkpoint일때
                if self.status == LONG:
                    close_func = self.close_long_limit
                elif self.status == SHORT:
                    close_func = self.close_short_limit


            while not self.order_filled():
                iter += 1
                time.sleep(self.time_interval)
                
                curr_pnl, profit = get_curr_pnl(self.binance, self.sym.replace("/", ""))
                price = self.inquire_curr_price()
                pre_pnls.append(curr_pnl)
                pre_prices.append(price)
                print(f"\r{iter} {self.sym.split('/')[0]} {status_str(self.status)}] PNL: {profit} ({pnlstr(round(curr_pnl, 1))})\t", end="")

                # 6시간 지나도 팔릴 가망이 없을 때

                # TODO: 가망이 생기면 원래대로 되돌려놔야해
                _6h = int(round(6*3600/self.time_interval))
                if curr_pnl < -self.anx_pnl and len(pre_pnls) > _6h and time.time() - pnl_lastupdate > 60*30:
                    new_order_price = pre_prices[np.argmax(pre_pnls[-_6h:])]
                    self.cancle_order()
                    close_func(price=new_order_price)
                    pnl_lastupdate = time.time()

            pop_from_existing_positions(self.status)
            print("\n!!! Limit Market Filled")
            balance = self.binance.fetch_balance()
            log_wallet_history(balance)
            return self.sym  # finish the iteration