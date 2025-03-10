import importlib
import argparse
import trader   # 파일 수정 반영
import time
import os
import datetime
import asyncio
from utils import pop_from_existing_positions
"""
TODO: 
1. 15m 따라서 숏/롱 포지션 바꾸기
2. 에러처리 try except를 while처리하기 
3. 점수를 socre화해서 188개중 "가장" 점수가 높은거 선택
"""



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
    parser.add_argument('--re_execution',
                        '-r',
                        default=True,
                        type=bool,
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
            if not args.re_execution:
                break
        except Exception as e: 
            print(e)
            time.sleep(1)