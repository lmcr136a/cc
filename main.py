import importlib
import argparse
import trader   # 파일 수정 반영
import time
import os
import datetime
import asyncio
from utils import pop_from_existing_positions



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol',
                        '-s',
                        default=None,
                        type=str,
                        )
    parser.add_argument('--number',
                        '-n',
                        default=1,
                        type=int,
                        )
    
    args = parser.parse_args()
    sym = args.symbol if not args.symbol or '/USDT' in args.symbol else args.symbol + '/USDT'
    
    while 1:
        try:
            importlib.reload(trader)
            minion = trader.Trader(sym, args.number)
            if minion.this_sym_is_running:
                continue
            before_sym, res = minion.run()
            now = datetime.datetime.now()
            now = now.strftime("%m%d_%H%M%S")
            if res in ["TP", "SL"]:
                os.rename(f"Figures/minion{args.number}.jpg", f"logs/{now}_{res}.jpg")
            sym = None
        except Exception as e: 
            print(e)
            time.sleep(1)