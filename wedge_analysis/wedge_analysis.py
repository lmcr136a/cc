from wedge_analysis.trendline_automation import *
from wedge_analysis.wedge import *
from utils import *
from ccxt.base.errors import BadSymbol
pd.set_option('mode.chained_assignment',  None)

async def realtime_analysis(sym, tf = "3m", limit = 60, imgfilename="realtime"):
    binance = get_binance()
    df = await past_data(binance, sym, tf, limit+7)
    df.reset_index(drop=False, inplace=True)
    df = add_RSI(df)
    df = df.iloc[-limit:]

    df["Pivot"] = 0
    df["Pivot"] = df.apply(lambda x: pivot_id(df, x.name, 4, 4), axis=1)
    df["Pivot"].iloc[-1] = 0
    df['PointPos'] = df.apply(lambda x: pivot_point_position(x), axis=1) # Used for visualising the pivot points

    df.reset_index(drop=True, inplace=True)
    await binance.close()
    
    pattern = None
    ent_price1 = None
    close_price1 = None
    ent_price2 = None
    close_price2 = None
    curr_idx = len(df["close"])-1
    curr_price = df["close"].iloc[-1]
    
    ## Plot
    plt.rcParams["figure.figsize"] = (6,6)
    plt.subplots_adjust(top = 0.95, bottom = 0.05, right = 0.98, left = 0.05, 
                hspace = 1, wspace = 1)
    f, (ax, ax_vol, ax_rsi) = plt.subplots(3, 1, gridspec_kw={'height_ratios': [4, 1, 1]})
    ax.set_facecolor((0.9, 0.9, 0.85))
    
    df["Index"] = df.index
    candlestick_ohlc(ax, df.loc[:, ["Index", "open", "high", "low", "close"] ].values, width=0.6, colorup='green', colordown='red', alpha=0.8)
    ax.scatter(df["Index"], df["PointPos"], color="b", s=20)
    ax.set_title(sym, position = (0.5,1.05),fontsize = 23)
    ax_vol.plot(df["volume"])
    ax_vol.set_title('Vol',position = (0.5,1.05),fontsize = 23)
    ax_rsi.plot(df["rsi"])
    ax_rsi.set_title('RSI',position = (0.5,1.05),fontsize = 23)
    ax_rsi.axhline(30, ls = '--', c='y', alpha = 0.9)
    ax_rsi.axhline(70, ls = '--', c='y', alpha = 0.9)

    ## graph analysis
    s_coefs, r_coefs = find_t_convergence(df['PointPos'], df['Pivot'], 3, df['close'][0])
    if not s_coefs:
        print('No convergence')
        plt.savefig(f"Figures/{imgfilename}.jpg", dpi = 300)
        plt.close()
        return None
    
    
    # r_rsis = np.array([df["rsi"].iloc[r_coefs["last_x1"]], df["rsi"].iloc[r_coefs["last_x2"]], df["rsi"].iloc[r_coefs["last_x3"]]])
    # s_rsis = np.array([df["rsi"].iloc[s_coefs["last_x1"]], df["rsi"].iloc[s_coefs["last_x2"]], df["rsi"].iloc[s_coefs["last_x3"]]])
    
    r_vols = np.array([df["volume"].iloc[r_coefs["last_x1"]], df["volume"].iloc[r_coefs["last_x2"]], df["volume"].iloc[r_coefs["last_x3"]]])
    s_vols = np.array([df["volume"].iloc[s_coefs["last_x1"]], df["volume"].iloc[s_coefs["last_x2"]], df["volume"].iloc[s_coefs["last_x3"]]])

    # d_r_rsi = np.sum(r_rsis[1:] - r_rsis[:-1])
    # d_s_rsi = np.sum(s_rsis[1:] - s_rsis[:-1])
    d_r_vol = True if r_vols[0] > r_vols[1] and r_vols[1] > r_vols[2] else False
    d_s_vol = True if s_vols[0] > s_vols[1] and s_vols[1] > s_vols[2] else False
    
    # ax_rsi.scatter([r_coefs["last_x1"],r_coefs["last_x2"],r_coefs["last_x3"]], r_rsis)
    ax_vol.scatter([r_coefs["last_x1"],r_coefs["last_x2"],r_coefs["last_x3"]], r_vols)
    # ax_rsi.scatter([s_coefs["last_x1"],s_coefs["last_x2"],s_coefs["last_x3"]], s_rsis)
    ax_vol.scatter([s_coefs["last_x1"],s_coefs["last_x2"],s_coefs["last_x3"]], s_vols)
    # if np.max(r_rsis) < 75 and np.min(s_rsis) > 25:
    #     print("No high/low RSI")
    #     return None
    
    # if (r_coefs["a"] > 0) and \
    # d_r_rsi < 0 and d_r_vol < 0:
    #     pattern = "Rising Wedge"
    # elif (s_coefs["a"] < 0 ) and \
    # d_s_rsi > 0 and d_s_vol < 0:
    #     pattern = "Falling Wedge"
        
    if 1:
        if d_s_vol or d_r_vol:
            pattern = "Triangular Convergence"
            
        x_tail_r = curr_idx - r_coefs["last_x3"]
        x_tail_s = curr_idx - s_coefs["last_x3"]
        support_line = s_coefs["a"] * np.arange(s_coefs["last_x3"]-s_coefs["start_x"] + x_tail_s) + s_coefs["b"]
        resist_line = r_coefs["a"] * np.arange(r_coefs["last_x3"]-r_coefs["start_x"] + x_tail_r) + r_coefs["b"]
        
        _support_line = [0]*s_coefs["last_x1"] + list(support_line)
        _resist_line = [0]*r_coefs["last_x1"] + list(resist_line)
        
        ent_price1 = _resist_line[-1]*(1+0.01*0.2)
        close_price1 = ent_price1+(_resist_line[r_coefs["last_x1"]]-_support_line[s_coefs["last_x1"]])
        ent_price2 = _support_line[-1]*(1-0.01*0.2)
        close_price2 = ent_price2-(_resist_line[r_coefs["last_x1"]]-_support_line[s_coefs["last_x1"]])
        
    if not pattern:
        print("No pattern")
        plt.savefig(f"Figures/{imgfilename}.jpg", dpi = 300)
        plt.close()
        return None
        
    
    s_x = np.arange(s_coefs["start_x"], curr_idx)
    r_x = np.arange(r_coefs["start_x"], curr_idx)

    ax.plot(s_x, support_line, color="b", linewidth=0.5, alpha=0.5)
    ax.plot(r_x, resist_line, color="b", linewidth=0.5, alpha=0.5)
    
    ax.scatter(curr_idx, ent_price1, c='b', alpha = 0.9, s=100, marker="s")
    ax.scatter(curr_idx, close_price1, c='r', alpha = 0.9, s=100, marker="x")
    if "Triangular" in pattern:
        ax.scatter(curr_idx, ent_price2, c='b', alpha = 0.9, s=100, marker="s")
        ax.scatter(curr_idx, close_price2, c='r', alpha = 0.9, s=100, marker="x")


    os.makedirs("Figures/", exist_ok=True)
    plt.savefig(f"Figures/{imgfilename}.jpg", dpi = 300)
    plt.close()
    
    if not(support_line[-1] < curr_price and resist_line[-1] > curr_price):
        print("Not now")
        return None
    
    # if np.max(r_rsis) < 75 and np.min(s_rsis) > 25:
    #     print("No high/low RSI")
    #     return None
    
    print(sym, pattern)
    return {
            "pattern": pattern,
            "ent_price1": ent_price1,
            "position1": LONG,
            "close_price1": close_price1,
            "ent_price2": ent_price2,
            "position2": SHORT,
            "close_price2": close_price2,
        }
    

async def select_sym(N):
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
    while return_pos is None:
        random.shuffle(symlist)
        for i, sym in enumerate(symlist):  # 0705 0.55초 걸
            # sym = '1000PEPE/USDT'
            try:
                vol = await binance.fetch_tickers(symbols=[sym])
                time.sleep(1)
                await binance.close()
                    
                if (not len(list(vol.values())) > 0) or list(vol.values())[0]['quoteVolume'] < 20*(10**6):
                    symlist.pop(i)
                    continue
                
                print(f"[{i}/{len(symlist)}]", sym, end="\t")
                res = await realtime_analysis(sym, imgfilename="minion"+str(N))
                
                if res:
                    print(sym)
                    print(res)
                    return sym, res
                
            except:
                symlist.pop(i)
                continue
        with open("symlist.txt", "w") as f:
            f.write(str(symlist))