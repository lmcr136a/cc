# from wedge_analysis.wedge_analysis import *
from ccxt.base.errors import BadSymbol
# from heikin_ashi.find_heikin import find_heikin
from wedge_analysis.support_resistance import find_support_resistance
# from ema_arr.find_ema_arr import find_ema_arrangement
# from ema_arr.find_ema_arr_opposite import find_ema_arrangement_opposite
# from fibonacci.find_fibonacci import find_fibonacci
from utils import *
pd.set_option('mode.chained_assignment',  None)

async def select_sym(N, tp):
    binance = get_binance()
    print(y("\nSEARCHING..."))
    return_pos = None
    
    await binance.load_markets()
    market = binance.markets
    symlist = []
    for s in market.keys():
        if s.split(":")[0][-4:] == "USDT":
            symlist.append(s.split(":")[0])
    symlist = list(set(symlist))
    # symlist = SYMLIST
    await binance.close()
    
    while return_pos is None:
        random.shuffle(symlist)
        for i, sym in enumerate(symlist):  # 0705 0.55초 걸
            if sym in ["USDC/USDT", "BTC/USDT"]:
                continue
            # sym = 'VANA/USDT'
            try:
                binance = get_binance()
                vol = await binance.fetch_tickers(symbols=[sym])
                time.sleep(1)
                await binance.close()
                    
                if (not len(list(vol.values())) > 0) or list(vol.values())[0]['quoteVolume'] < 1*(10**5):
                    symlist.pop(i)
                    continue
                
                print(f"[{i}/{len(symlist)}]", sym)
                res = await find_support_resistance(sym, tp, imgfilename="minion"+str(N))
                # res = await find_wedge(sym, pnl, imgfilename="minion"+str(N))
                # if not res:
                # res = await find_heikin(sym, imgfilename="minion"+str(N))
                
                if res:
                    print(sym)
                    print(res)
                    return sym, res
                
            except BadSymbol:
                await binance.close()
                symlist.pop(i)
                continue
        with open("symlist.txt", "w") as f:
            f.write(str(symlist))