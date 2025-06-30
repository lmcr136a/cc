import importlib
import numpy as np
import time
import utils
from utils import *
from select_sym import select_sym
# from ema_arr.find_ema_arr import tracking
# from ema_arr.find_ema_arr_opposite import tracking
from wedge_analysis.support_resistance import tracking
import asyncio



class Trader():
    def __init__(self, symbol=None, number=1) -> None:
        self.N = int(number)
        self.init_amt = 0
        self.other_running_sym_num = 0
        self.status = None
        self.lev = 1                # 0.04*lev 가 수수료
        self.sl = -5*self.lev       # 마이너스인거 확인 & 지금 안쓰임
        self.tp = 1*self.lev        # 지금 안쓰임
        self.limit_amt_ratio = 0.0003
        
        self.already_running = False
        if not symbol:
            self.sym, self.res = asyncio.run(select_sym(number, self.tp/self.lev))
            self.checkpoint=False
        else:
           self.sym = symbol
           self.checkpoint=True

        self.time_interval = 3
        self.this_sym_is_running = asyncio.run(self.inquire_curr_info(init=True))
        
        if not self.init_amt:  # 기존꺼 가져오는 경우가 아니라 걍 첨에 시작하는 경우
            self.amount = cal_compound_amt(self.lev, float(self.price))
            self.close_order_id = None
        else:
            if not symbol:
                pass
            else: 
                self.close_order_id = asyncio.run(self.get_open_order_id())
        print(f"{'*'*50}\nwallet:{round(self.wallet_usdt, 3)}  lev:{self.lev} stop loss: {self.sl}")

    
    async def get_open_order_id(self,):
        self.binance = get_binance()
        open_order_info = await self.binance.fetch_open_orders(symbol=self.sym)
        await self.binance.close()
        return open_order_info[0]['info']['orderId']

    def update_wallet(self, balance=None):
        if not balance:
            balance = self.binance.fetch_balance()
        # USDT
        for asset in balance['info']['assets']:
            if asset['asset'] == 'USDT':
                self.wallet_usdt = float(asset['availableBalance'])

    ## 현재 정보 조회
    async def inquire_curr_info(self, init=False):
        self.binance = get_binance()
        await self.binance.fetch_markets()
        balance = await self.binance.fetch_balance(params={"type": "future"})
        info = await self.binance.fetch_ticker(self.sym)
        self.price = info['close']
        await self.binance.close()
        positions = balance['info']['positions']
        pose_info = None
        
        log_wallet_history(balance)
        # USDT
        self.update_wallet(balance)
        for position in positions:
            amt = float(position['positionAmt'])
            if abs(amt) > 0 and position["symbol"] != self.sym.replace("/", ""):
                self.other_running_sym_num += 1
                
            if abs(amt) > 0 and position["symbol"] == self.sym.replace("/", ""):
                return True
            
        print(f"other_running_sym_num: {self.other_running_sym_num}")

        if init:
            await self.init_ckpt()

    async def get_curr_sym_amt(self):
        self.binance = get_binance()
        await self.binance.fetch_markets()
        balance = await self.binance.fetch_balance(params={"type": "future"})
        await self.binance.close()
        positions = balance['info']['positions']
        for position in positions:
            if position["symbol"] == self.sym.replace("/", ""):
                amt = float(position['positionAmt'])
                return amt
        return 0
        

    async def init_ckpt(self):
        binance = get_binance()
        positions = await binance.fetch_balance()
        amt = 0
        pnl = 0
        self.entry_price = 0
        for pos in positions['info']['positions']:
            if pos['symbol'] == self.sym.replace("/", ""):
                print(f'Init checkpoint {self.sym}')
                amt = float(pos['positionAmt'])
                profit = float(pos['unrealizedProfit'])
                pnl = float(pos['unrealizedProfit'])/float(pos['initialMargin'])*100
                break
        if amt < 0:
            self.status = SHORT
        elif amt > 0:
            self.status = LONG
        self.amount = abs(amt)
        self.init_amt = amt
        await binance.close()
        print(f"{self.sym} {self.status} Current PNL:", pnl, "% Leverage: ", self.lev, ' entry_price: ', self.entry_price)
        
    
    async def get_curr_price(self,):
        binance = get_binance()
        ticker = await binance.fetch_ticker(self.sym)
        await binance.close()
        return ticker['last']


    async def prep_order(self,):
        raw_price = await self.get_curr_price()
        # point_len = len(str(raw_price).split(".")[-1])
        # price = round(raw_price*(1-self.limit_amt_ratio), point_len)  
        price = raw_price if not self.ent_price else self.ent_price

        binance = get_binance()
        await binance.load_markets()
        symli = list(binance.markets.keys())
        if self.sym in symli or self.sym+":USDT" in symli:
            try:
                market = binance.markets[self.sym]
            except:
                market = binance.markets[self.sym+":USDT"]
                
            resp = await binance.set_leverage(
                symbol=market['id'],
                leverage=self.lev,
            )
        else:
            raise
        await binance.close()
        print()
        print(f"SUBMITTIED ORDER [{self.sym}] price:{price}")
        return price

    async def open_order(self):
        self.binance = get_binance()
        self.entry_price = await self.prep_order()
        
        if self.status == LONG:
            side = "buy"
        elif self.status == SHORT:
            side = "sell"
            
        order = await self.binance.create_order(
            symbol=self.sym, type="limit", side=side, amount=np.abs(self.amount), price=self.entry_price)
        
        self.order_id = order['id']
        await self.binance.close()
        
    async def close_market_order(self):
        self.binance = get_binance()
        if self.status == LONG:
            side = "sell"
        elif self.status == SHORT:
            side = "buy"
        order = await self.binance.create_order(
            symbol=self.sym, type="market", side=side, amount=np.abs(self.amount))
        # self.order_id = order['id']
        await self.binance.close()
        
    async def open_market_order(self):
        self.binance = get_binance()
        await self.prep_order()
        
        if self.status == LONG:
            side = "buy"
        elif self.status == SHORT:
            side = "sell"
        order = await self.binance.create_order(
            symbol=self.sym, type="market", side=side, amount=np.abs(self.amount))
        # self.order_id = order['id']
        await self.binance.close()
        
    async def close_limit_order(self, entry_price=None):
        self.binance = get_binance()
        if not entry_price:
            entry_price = await self.prep_order()
        if self.status == LONG:
            close_side = "sell"
            # self.tp_price = entry_price*(1+self.takeprofit*0.01/self.lev)
        elif self.status == SHORT:
            close_side = "buy"
            # self.tp_price = entry_price*(1-self.takeprofit*0.01/self.lev)
            
        close_order = await self.binance.create_order(
            symbol=self.sym, type="limit", side=close_side, amount=np.abs(self.amount), price=self.tp_price)
        self.close_order_id = close_order['id']
        await self.binance.close()
        
    async def cancel_order(self, order_id):
        self.binance = get_binance()
        response = await self.binance.cancel_order(id=int(order_id), symbol=self.sym)
        await self.binance.close()
        return response


    def run(self):
        iter = 0
        self.pre_pnls = []
        self.ent_price = None
        open_to_buy_more = True
        self.t = 0
        while 1:
            
            if not self.status:
                curr_price = asyncio.run(self.get_curr_price())
                if self.res["ent_price1"] and (curr_price >= self.res["ent_price1"] or self.res["curr_price1"]):
                    self.position_to_by = self.res["position1"]
                    self.tp_price = self.res["tp_price1"]
                    self.sl_price = self.res["sl_price1"]
                    self.ent_price = curr_price if self.res['curr_price1'] else self.res['ent_price1']
                    self.sl = (self.res['sl_price1'] - self.ent_price)/self.ent_price*100*self.lev
                    self.tp = (self.res['tp_price1'] - self.ent_price)/self.ent_price*100*self.lev

                elif self.res["ent_price2"] and (curr_price <= self.res["ent_price2"] or self.res["curr_price2"]):
                    self.position_to_by = self.res["position2"]
                    self.sl_price = self.res["sl_price2"]
                    self.tp_price = self.res["tp_price2"]
                    self.ent_price = curr_price if self.res['curr_price2'] else self.res['ent_price2']
                    self.sl = -(self.res['sl_price2'] - self.ent_price)/self.ent_price*100*self.lev
                    self.tp = -(self.res['tp_price2'] - self.ent_price)/self.ent_price*100*self.lev
                
                # else:
                #     max_waiting = 1*60/self.time_interval
                #     print(f"\rWaiting.. {self.t} / {max_waiting}", end="")
                #     if self.t > max_waiting:  
                #         asyncio.run(self.cancel_order(self.order_id))
                #         return self.sym, "X_buy"
                    
                #     self.t += 1 
                #     time.sleep(self.time_interval)
                #     continue
                
                self.status = self.position_to_by
                asyncio.run(self.open_market_order())
                
            else :
                curr_amt = asyncio.run(self.get_curr_sym_amt())
                
                if not curr_amt:
                    max_waiting = 5*60/self.time_interval
                    print(f"\rWaiting.. {self.t} / {max_waiting}", end="")
                    if self.t > max_waiting:  # limit order 안사짐
                        asyncio.run(self.cancel_order(self.order_id))
                        return self.sym, "X_buy"
                        
                elif curr_amt:
                    curr_pnl, profit = asyncio.run(get_curr_pnl(self.sym.replace("/", "")))
                    self.tp_price, self.sl_price, tp_close, sl_close, buy_more = asyncio.run(tracking(sym=self.sym, ent_price=self.ent_price, 
                                                                                                      sl_price=self.sl_price, tp_price=self.tp_price, 
                                                                                                      open_to_buy_more=open_to_buy_more, position=self.status, imgfilename=f"minion{self.N}"))
                    # if self.t > 500 and curr_pnl > 0:
                    #     tp_close = True
                        
                    if self.status == LONG:
                        self.sl = (self.sl_price - self.ent_price)/self.ent_price*100*self.lev
                        self.tp = (self.tp_price - self.ent_price)/self.ent_price*100*self.lev
                    else:
                        self.sl = -(self.sl_price - self.ent_price)/self.ent_price*100*self.lev
                        self.tp = -(self.tp_price - self.ent_price)/self.ent_price*100*self.lev
                    self.pre_pnls.append(curr_pnl)
                    
                    if tp_close or sl_close:
                        asyncio.run(self.close_market_order())
                        # asyncio.run(self.cancel_order(self.close_order_id))
                        if tp_close:
                            print("===Take Profit===")
                            return self.sym, "TP"  # finish the iteration
                        else:
                            print("===Stop Loss===")
                            return self.sym, "SL"  # finish the iteration
                    # elif buy_more:
                    #     asyncio.run(self.open_market_order())
                    #     open_to_buy_more = False
                    #     self.amount = np.abs(float(curr_amt))
                    #     self.ent_price = np.mean([self.ent_price, buy_more])
                
                    print(f"\r{self.N}[{self.sym.split('/')[0]}_{status_str(self.status)}]_PNL:{profit}({pnlstr(round(curr_pnl, 2))})__SL:{pnlstr(round(self.sl, 2))}_TP:{pnlstr(round(self.tp, 2))}__{round(self.t*self.time_interval/60)}min  ", end="")
                else:
                    print(f"\r{self.N}[{self.sym.split('/')[0]}_{status_str(self.status)}]_Waiting__SL:{pnlstr(round(self.sl, 2))}_TP:{pnlstr(round(self.tp, 2))}__{round(self.t*self.time_interval/60)}min  ", end="")
                self.t += 1 

            time.sleep(self.time_interval)
            iter += 1
