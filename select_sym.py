from wedge_analysis.wedge_analysis import *
from heikin_ashi.find_heikin import find_heikin
from utils import *
pd.set_option('mode.chained_assignment',  None)

async def select_sym(N):
    binance = get_binance()
    print(y("\nSEARCHING..."))
    return_pos = None
    
    await binance.load_markets()
    market = binance.markets
    # symlist = []
    # for s in market.keys():
    #     if s.split(":")[0][-4:] == "USDT":
    #         symlist.append(s.split(":")[0])
    # symlist = list(set(symlist))
    symlist = SYMLIST
    while return_pos is None:
        random.shuffle(symlist)
        for i, sym in enumerate(symlist):  # 0705 0.55초 걸
            # sym = 'ETC/USDT'
            try:
                vol = await binance.fetch_tickers(symbols=[sym])
                time.sleep(1)
                await binance.close()
                    
                if (not len(list(vol.values())) > 0) or list(vol.values())[0]['quoteVolume'] < 20*(10**7):
                    symlist.pop(i)
                    continue
                
                print(f"[{i}/{len(symlist)}]", sym, end="\t")
                res = await find_wedge(sym, imgfilename="minion"+str(N))
                # if not res:
                # res = await find_heikin(sym, imgfilename="minion"+str(N))
                
                if res:
                    print(sym)
                    print(res)
                    return sym, res
                
            except BadSymbol:
                symlist.pop(i)
                continue
        with open("symlist.txt", "w") as f:
            f.write(str(symlist))