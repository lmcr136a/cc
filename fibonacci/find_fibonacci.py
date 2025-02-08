from wedge_analysis.trendline_automation import *
from wedge_analysis.wedge import *
from utils import *
from HYPERPARAMETERS import *
pd.set_option('mode.chained_assignment',  None)



def get_fibo_prices(df):
    high, high_i = np.max(df["high"]), np.argmax(df["high"])
    low, low_i = min(df["low"]), np.argmin(df["low"])
    if low_i < high_i:
        position = LONG
        p1 = high - 0.618*(high-low)
        p2 = high - 0.5*(high-low)
        p3 = high - 0.382*(high-low)
    else:
        position = SHORT
        p1 = low - 0.382*(low-high)
        p2 = low - 0.5*(low-high)
        p3 = low - 0.618*(low-high)
    return position, low, p1, p2, p3, high


async def find_fibonacci(sym, pnl, tf = "5m", limit = 150, n=5, imgfilename="realtime", decided_res=None):
    pnl *= 0.01
    binance = get_binance()
    df = await past_data(binance, sym, tf, limit+n)
    
    df.reset_index(drop=False, inplace=True)
    df = df.iloc[-limit:]

    # df["Pivot"] = 0
    # df["Pivot"] = df.apply(lambda x: pivot_id(df, x.name, limit, limit), axis=1)
    # df["Pivot"].iloc[-1] = 0
    # df['PointPos'] = df.apply(lambda x: pivot_point_position(x), axis=1) # Used for visualising the pivot points

    # df.reset_index(drop=True, inplace=True)
    await binance.close()
    
    pattern = None
    ent_price1, close_price1, stop_price1, curr_price1 = None, None, None, False
    ent_price2, close_price2, stop_price2, curr_price2 = None, None, None, False
    curr_idx = len(df["close"])-1
    curr_price = df["close"].iloc[-1]
    
    df["Index"] = df.index
    
    ## Plot
    plot=False
    if plot:
        plt.rcParams["figure.figsize"] = (6,4)
        f, ax = plt.subplots(1,1)
        ax.set_facecolor((0.9, 0.9, 0.85))
        plt.subplots_adjust(top = 0.9, bottom = 0.05, right = 0.98, left = 0.1, 
                    hspace = 0.4, wspace = 0.4)
    
        candlestick_ohlc(ax, df.loc[:, ["Index", "open", "high", "low", "close"] ].values, width=0.6, colorup='green', colordown='red', alpha=0.8)
    
    ## graph analysis
    if np.argmax(df["high"]) in [limit-1, limit-2] or np.argmax(df["close"]) in [limit-1, limit-2]:
        pattern = "Bull"
        ent_price1 = curr_price
        close_price1 = curr_price*(1+pnl)
        curr_price1 = True
        # ent_price2 = curr_price
        # close_price2 = curr_price*(1-pnl)
        # curr_price2 = True
    elif np.argmin(df["low"]) in list(range(limit-3, limit)) or np.argmin(df["close"]) in list(range(limit-3, limit)):
        pattern = "Bear"
        ent_price2 = curr_price
        close_price2 = curr_price*(1-pnl)
        curr_price2 = True
                
    elif np.argmax(df["high"]) in list(range(10)) or np.argmin(df["low"]) in list(range(10)):
        if plot:
            ax.set_title(f"Nothing - {sym}, {tf}", position = (0.5,1.05),fontsize = 18)
            plt.savefig(f"Figures/{imgfilename}.jpg", dpi = 300)
            plt.close()
        return decided_res
    
    position, low, p1, p2, p3, high = get_fibo_prices(df)
        
    # if position == LONG and curr_price < p2 and curr_price > p1:
    #     pattern = "LONG Fibo"
    #     ent_price1 = curr_price
    #     close_price1 = p3
    #     stop_price1 = p1*(1-0.01)
    #     curr_price1 = True
    # elif position == SHORT and curr_price > p2 and curr_price < p3:
    #     pattern = "SHORT Fibo"
    #     ent_price2 = curr_price
    #     close_price2 = p1
    #     stop_price2 = p3*(1+0.01)
    #     curr_price2 = True
        
    
    if not pattern:
        if plot:
            plt.savefig(f"Figures/{imgfilename}.jpg", dpi = 300)
            plt.close()
        return decided_res
        
    
    if plot:
        ax.axhline(low, ls = '--', color=(0.1, 0.5, 0.9), alpha = 0.5)
        ax.axhline(p1, ls = '--', color=(0.3, 0.5, 0.7), alpha = 0.5)
        ax.axhline(p2, ls = '--', color=(0.5, 0.5, 0.5), alpha = 0.5)
        ax.axhline(p3, ls = '--', color=(0.7, 0.5, 0.3), alpha = 0.5)
        ax.axhline(high, ls = '--', color=(0.9, 0.5, 0.1), alpha = 0.5)
        ax.set_title(f"{pattern} - {sym}, {tf}", position = (0.5,1.05),fontsize = 18)
        
        if ent_price1:
            ax.scatter(curr_idx, ent_price1, color=BUY_COLOR, alpha = 0.6, s=50, marker="s")
            ax.scatter(curr_idx, close_price1, color=BUY_COLOR, alpha = 0.6, s=50, marker="x")
            
        if ent_price2:
            ax.scatter(curr_idx, ent_price2, color=SELL_COLOR, alpha = 0.6, s=50, marker="s")
            ax.scatter(curr_idx, close_price2, color=SELL_COLOR, alpha = 0.6, s=50, marker="x")

        os.makedirs("Figures/", exist_ok=True)
        plt.savefig(f"Figures/{imgfilename}.jpg", dpi = 300)
        plt.close()
        
    return {
            "pattern": pattern,
            "ent_price1": ent_price1,
            "position1": LONG,
            "close_price1": close_price1,
            "stop_price1": stop_price1,
            "curr_price1": curr_price1,
            
            "ent_price2": ent_price2,
            "position2": SHORT,
            "close_price2": close_price2,
            "stop_price2": stop_price2,
            "curr_price2": curr_price2,
        }
    