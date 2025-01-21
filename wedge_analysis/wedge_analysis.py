from wedge_analysis.trendline_automation import *
from wedge_analysis.wedge import *
from utils import *
from cal_utils import cal_rsi, cal_srsi
from ccxt.base.errors import BadSymbol
from HYPERPARAMETERS import *
pd.set_option('mode.chained_assignment',  None)

async def find_wedge(sym, tf = "1m", limit = 80, n=5, imgfilename="realtime", decided_res=None):
    binance = get_binance()
    df = await past_data(binance, sym, tf, limit+n)
    df['ema200'] = df['close'].ewm(50).mean()
    
    df = df.iloc[-limit-n:]
    srsi_k, srsi_d = cal_srsi(df["close"], n=n)
    df['srsi_k'] = srsi_k
    df['srsi_d'] = srsi_d
    
    
    df.reset_index(drop=False, inplace=True)
    df = df.iloc[-limit:]

    df["Pivot"] = 0
    df["Pivot"] = df.apply(lambda x: pivot_id(df, x.name, 3, 3), axis=1)
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
    f, (ax, ax_vol, ax_rsi) = plt.subplots(3, 1, gridspec_kw={'height_ratios': [4, 1, 1]})
    ax.set_facecolor((0.9, 0.9, 0.85))
    plt.subplots_adjust(top = 0.95, bottom = 0.05, right = 0.98, left = 0.05, 
                hspace = 1, wspace = 1)
    
    df["Index"] = df.index
    candlestick_ohlc(ax, df.loc[:, ["Index", "open", "high", "low", "close"] ].values, width=0.6, colorup='green', colordown='red', alpha=0.8)
    ax.scatter(df["Index"], df["PointPos"], color="b", s=20)
    ax.set_title(f"Convergence - {sym}, {tf}", position = (0.5,1.05),fontsize = 23)
    # ax.set_title(sym, position = (0.5,1.05),fontsize = 23)
    ax_vol.plot(df["volume"])
    ax_vol.set_title('Vol',position = (0.5,1.05),fontsize = 23)
    
    ax_rsi.plot(df["srsi_k"], color=(0.2, 0.3, 0.8))
    ax_rsi.plot(df["srsi_d"], color=(0.9, 0.6, 0.1))
    ax_rsi.set_title('S-RSI',position = (0.5,1.05),fontsize = 23)
    ax_rsi.axhline(25, ls = '--', c='y', alpha = 0.9)
    ax_rsi.axhline(75, ls = '--', c='y', alpha = 0.9)
    ## graph analysis
    s_coefs, r_coefs = find_t_convergence(df['PointPos'], df['Pivot'], 3, df['close'][0])
    if not s_coefs:
        print('No convergence')
        plt.savefig(f"Figures/{imgfilename}.jpg", dpi = 300)
        plt.close()
        return decided_res
    
    
    # r_rsis = np.array([df["rsi"].iloc[r_coefs["last_x1"]], df["rsi"].iloc[r_coefs["last_x2"]], df["rsi"].iloc[r_coefs["last_x3"]]])
    # s_rsis = np.array([df["rsi"].iloc[s_coefs["last_x1"]], df["rsi"].iloc[s_coefs["last_x2"]], df["rsi"].iloc[s_coefs["last_x3"]]])
    
    r_vols = np.array([df["volume"].iloc[r_coefs["last_x1"]], df["volume"].iloc[r_coefs["last_x2"]], df["volume"].iloc[r_coefs["last_x3"]]])
    s_vols = np.array([df["volume"].iloc[s_coefs["last_x1"]], df["volume"].iloc[s_coefs["last_x2"]], df["volume"].iloc[s_coefs["last_x3"]]])

    # d_r_rsi = np.sum(r_rsis[1:] - r_rsis[:-1])
    # d_s_rsi = np.sum(s_rsis[1:] - s_rsis[:-1])
    # d_r_vol = True if r_vols[0] > r_vols[1] and r_vols[1] > r_vols[2] else False
    # d_s_vol = True if s_vols[0] > s_vols[1] and s_vols[1] > s_vols[2] else False
    
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
        
    pnl = 0.005
    if 1:
        # if d_s_vol or d_r_vol:
        pattern = "Triangular Convergence"
            
        x_tail_r = curr_idx - r_coefs["last_x3"]
        x_tail_s = curr_idx - s_coefs["last_x3"]
        support_line = s_coefs["a"] * np.arange(s_coefs["last_x3"]-s_coefs["start_x"] + x_tail_s) + s_coefs["b"]
        resist_line = r_coefs["a"] * np.arange(r_coefs["last_x3"]-r_coefs["start_x"] + x_tail_r) + r_coefs["b"]
        
        _support_line = [0]*s_coefs["last_x1"] + list(support_line)
        _resist_line = [0]*r_coefs["last_x1"] + list(resist_line)
        
        ent_price1 = _resist_line[-1]#*(1+0.01*0.2)
        close_price1 = pnl*ent_price1 + ent_price1
        # close_price1 = ent_price1+(_resist_line[r_coefs["last_x1"]]-_support_line[s_coefs["last_x1"]])
        ent_price2 = _support_line[-1]#*(1-0.01*0.2)
        close_price2 = -pnl*ent_price2 + ent_price2
        # close_price2 = ent_price2-(_resist_line[r_coefs["last_x1"]]-_support_line[s_coefs["last_x1"]])
        
    if not pattern:
        print("No pattern")
        plt.savefig(f"Figures/{imgfilename}.jpg", dpi = 300)
        plt.close()
        return decided_res
        
    
    s_x = np.arange(s_coefs["start_x"], curr_idx)
    r_x = np.arange(r_coefs["start_x"], curr_idx)

    ax.plot(s_x, support_line, color="b", linewidth=0.5, alpha=0.5)
    ax.plot(r_x, resist_line, color="b", linewidth=0.5, alpha=0.5)
     
    ax.scatter(curr_idx, ent_price1, color=BUY_COLOR, alpha = 0.6, s=50, marker="s")
    ax.scatter(curr_idx, close_price1, color=BUY_COLOR, alpha = 0.6, s=50, marker="x")
    if "Triangular" in pattern:
        ax.scatter(curr_idx, ent_price2, color=SELL_COLOR, alpha = 0.6, s=50, marker="s")
        ax.scatter(curr_idx, close_price2, color=SELL_COLOR, alpha = 0.6, s=50, marker="x")


    os.makedirs("Figures/", exist_ok=True)
    plt.savefig(f"Figures/{imgfilename}.jpg", dpi = 300)
    plt.close()
    
    if decided_res:
        if support_line[-1] > curr_price: 
            ent_price1 = None
            close_price1 = None
            ent_price2 = curr_price
            close_price2 = -pnl*ent_price2 + ent_price2
            
        elif resist_line[-1] < curr_price:
            ent_price1 = curr_price
            close_price1 = pnl*ent_price1 + ent_price1
            ent_price2 = None
            close_price2 = None
    else:
        if support_line[-1] > curr_price or resist_line[-1] < curr_price:
            print("Not now")
            return None
    # if np.max(r_rsis) < 75 and np.min(s_rsis) > 25:
    #     print("No high/low RSI")
    #     return None
    
    # print(sym, pattern)
    return {
            "pattern": pattern,
            "ent_price1": ent_price1,
            "position1": LONG,
            "close_price1": close_price1,
            "ent_price2": ent_price2,
            "position2": SHORT,
            "close_price2": close_price2,
        }
    