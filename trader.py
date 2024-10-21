import importlib
import numpy as np
import time
import utils
from utils import *
import asyncio

"""
1.2
"""
class Trader():
    def __init__(self, symbol=None, symnum=1) -> None:
        self.symnum = float(symnum)
        self.other_running_sym_num = 0
        self.lev = 2  # 0.04*lev 가 수수료
        self.stoploss = -30                     # 마이너스인거 확인
        self.takeprofit = 0.3   # 3 * 10
        self.limit_amt_ratio = 0.0003
        
        if not symbol:
            self.sym, self.position_to_by = asyncio.run(select_sym(self.symnum))
            self.checkpoint=False
        else:
           self.sym = symbol
           self.checkpoint=True

        self.time_interval = 5
        self.status = None
        asyncio.run(self.inquire_curr_info(init=True))
        
        if not self.init_amt:  # 기존꺼 가져오는 경우가 아니라 걍 첨에 시작하는 경우
            self.amount = cal_compound_amt(self.wallet_usdt, self.lev, float(self.price), self.symnum)
            self.close_order_id = None
        else:
            if not symbol:
                pass
            else: 
                self.close_order_id = asyncio.run(self.get_open_order_id())
        print(f"{'*'*50}\nwallet:{round(self.wallet_usdt, 3)}  lev:{self.lev} [[{self.stoploss}~{self.takeprofit}]]")

    
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
            
            if position['symbol'] == self.sym.replace("/", ""):
                pose_info = position
                
        self.symnum -= self.other_running_sym_num 
        print(f"other_running_sym_num: {self.other_running_sym_num}  self.symnum: {self.symnum}")

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
        

    async def prep_order(self,):
        binance = get_binance()
        await binance.load_markets()
        ticker = await binance.fetch_ticker(self.sym)
        raw_price = ticker['last']
        # raw_price = info['close']
        point_len = len(str(raw_price).split(".")[-1])
        price = round(raw_price*(1-self.limit_amt_ratio), point_len)  

        market = binance.markets[self.sym]
        resp = await binance.set_leverage(
            symbol=market['id'],
            leverage=self.lev,
        )
        await binance.close()
        print(f"SUBMITTING ORDER [{self.sym}] price:{price} {self.status}")
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
        self.order_id = order['id']
        await self.binance.close()
        
        
    async def close_limit_order(self, entry_price=None):
        self.binance = get_binance()
        if not entry_price:
            entry_price = await self.prep_order()
        if self.status == LONG:
            close_side = "sell"
            self.close_price = entry_price*(1+self.takeprofit*0.01/self.lev)
        elif self.status == SHORT:
            close_side = "buy"
            self.close_price = entry_price*(1-self.takeprofit*0.01/self.lev)
            
        close_order = await self.binance.create_order(
            symbol=self.sym, type="limit", side=close_side, amount=np.abs(self.amount), price=self.close_price)
        self.close_order_id = close_order['id']
        await self.binance.close()
        
        
    async def cancel_order(self, order_id):
        self.binance = get_binance()
        await self.binance.cancel_order(id=int(order_id), symbol=self.sym)
        await self.binance.close()
        

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
                    asyncio.run(self.open_order())
                    
                except Exception as error:
                    print(error)
                    self.status = None
                    if "Leverage" in str(error):
                        self.lev = round(0.5*self.lev)

                if not self.status:
                    self.missed_timing += 1

            else :
                curr_amt = asyncio.run(self.get_curr_sym_amt())
                if len(self.pre_pnls) == 1000 and self.entry_price > 0:
                    self.takeprofit = 3
                    asyncio.run(self.cancel_order(self.close_order_id))
                    asyncio.run(self.close_limit_order(entry_price=self.entry_price))
                    
                if not self.close_order_id:
                    if len(self.pre_pnls) > 20 and not curr_amt:  # limit order 안사짐
                        asyncio.run(self.cancel_order(self.order_id))
                        return self.sym 
                    
                    if abs(curr_amt) > 0 and self.init_amt == 0:  # open limit close order
                        asyncio.run(self.close_limit_order())

                else:
                    if abs(curr_amt) == 0:
                        print("Take profit limit order filled ! ")
                        return self.sym
                    
                close_position, curr_pnl = timing_to_close(sym=self.sym, 
                                                           satisfying_profit=self.takeprofit, 
                                                           max_loss=self.stoploss)
                self.pre_pnls.append(curr_pnl)
                if curr_pnl == -100:
                    return self.sym
                
                if close_position:
                    try:
                        asyncio.run(self.cancel_order(self.close_order_id))
                        asyncio.run(self.close_market_order())
                        return self.sym  # finish the iteration
                    except Exception as error:
                        print(error)
                
                

            time.sleep(self.time_interval)
            iter += 1
