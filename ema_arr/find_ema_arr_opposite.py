import os
from mplfinance.original_flavor import candlestick_ohlc
from utils import *
from HYPERPARAMETERS import *
pd.set_option('mode.chained_assignment',  None)
GAP_CORR = 2
TF = "5m"

def get_cross_points(col, i, n=3):
    t_list = [i+x for x in range(-n, 1)]
    
    u_cross, d_cross = True, True
    try:
        for t in t_list:
            if col.loc[t] < 0:
                d_cross = False
            if col.loc[t] > 0:
                u_cross = False
        if d_cross and col.loc[i+1] < 0:
            return 2
        if u_cross and col.loc[i+1] > 0:
            return 1
    except KeyError as e:
        pass
    return 0

def get_ucross_point_val(row, n1, n2):
    if row[f"cross{n1}{n2}"] == 1:
        return np.mean([row[f"ema{n1}"], row[f"ema{n2}"]])
    return np.nan

def get_dcross_point_val(row, n1, n2):
    if row[f"cross{n1}{n2}"] == 2:
        return np.mean([row[f"ema{n1}"], row[f"ema{n2}"]])
    return np.nan


def add_emas(df):
    df['ema1'] = df['close'].ewm(span=3, adjust=False).mean()
    df['ema2'] = df['close'].ewm(span=6, adjust=False).mean()
    df['ema3'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema4'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema5'] = df['close'].ewm(span=25*4, adjust=False).mean()
    df['ema12'] = df['ema1']-df['ema2']
    df['ema23'] = df['ema2']-df['ema3']
    df['ema34'] = df['ema3']-df['ema4']
    
    df.reset_index(drop=False, inplace=True)
    
    df["cross12"] = df.apply(lambda x: get_cross_points(df['ema12'], x.name), axis=1)
    df["cross23"] = df.apply(lambda x: get_cross_points(df['ema23'], x.name), axis=1)
    df["cross34"] = df.apply(lambda x: get_cross_points(df['ema34'], x.name), axis=1)
    
    df["cross12_u"] = df.apply(lambda x: get_ucross_point_val(x, 1, 2), axis=1)
    df["cross12_d"] = df.apply(lambda x: get_dcross_point_val(x, 1, 2), axis=1)
    df["cross23_u"] = df.apply(lambda x: get_ucross_point_val(x, 2, 3), axis=1)
    df["cross23_d"] = df.apply(lambda x: get_dcross_point_val(x, 2, 3), axis=1)
    df["cross34_u"] = df.apply(lambda x: get_ucross_point_val(x, 3, 4), axis=1)
    df["cross34_d"] = df.apply(lambda x: get_dcross_point_val(x, 3, 4), axis=1)
    return df
    


async def find_ema_arrangement_opposite(sym, pnl, tf = TF, limit = 60, imgfilename="realtime", decided_res=None):
    pnl *= 0.01
    binance = get_binance()
    df = await past_data(binance, sym, tf, limit)
    df = add_emas(df)

    await binance.close()
    
    pattern = None
    ent_price1, tp_price1, sl_price1, curr_price1 = None, None, None, False
    ent_price2, tp_price2, sl_price2, curr_price2 = None, None, None, False
    curr_idx = len(df["close"])-1
    curr_price = df["close"].iloc[-1]
    df["Index"] = df.index
    
    ## Plot
    plot=True
    if plot:
        RED, ORANGE, YELLOW, GREEN = (0.6, 0, 0, 1), (0.7, 0.5, 0, 1), (0.6, 0.6, 0, 1), (0.1, 0.5, 0, 1)
        RED2, ORANGE2, YELLOW2 = (0.9, 0.3, 0, 1), (0.9, 0.7, 0, 1), (0.6, 0.9, 0, 1)
        plt.rcParams["figure.figsize"] = (6,4)
        f, ax = plt.subplots(1,1)
        ax.set_facecolor((0.95, 0.95, 0.9))
        plt.subplots_adjust(top = 0.9, bottom = 0.05, right = 0.98, left = 0.1, 
                    hspace = 0.4, wspace = 0.4)
        ax.plot(df["Index"], df["ema1"], color=RED)
        ax.plot(df["Index"], df["ema2"], color=ORANGE)
        ax.plot(df["Index"], df["ema3"], color=YELLOW)
        ax.plot(df["Index"], df["ema4"], color=GREEN)
        ax.plot(df["Index"], df["ema5"], color="k")
        candlestick_ohlc(ax, df.loc[:, ["Index", "open", "high", "low", "close"] ].values, width=0.6, colorup='green', colordown='red', alpha=0.8)
        markersize = 60
        ax.scatter(df["Index"], df["cross12_d"], color=RED2, marker="v", s=markersize, zorder=5)
        ax.scatter(df["Index"], df["cross12_u"], color=RED2, marker="^", s=markersize, zorder=5)
        ax.scatter(df["Index"], df["cross23_d"], color=ORANGE2, marker="v", s=markersize, zorder=5)
        ax.scatter(df["Index"], df["cross23_u"], color=ORANGE2, marker="^", s=markersize, zorder=5)
        ax.scatter(df["Index"], df["cross34_d"], color=YELLOW2, marker="v", s=markersize, zorder=5)
        ax.scatter(df["Index"], df["cross34_u"], color=YELLOW2, marker="^", s=markersize, zorder=5)
        
    i_ucross12, i_ucross23, i_ucross34 = -1, -1, -1
    i_dcross12, i_dcross23, i_dcross34 = -1, -1, -1
    num_all_cross = 0
    for i in range(curr_idx):
        if df["cross12_u"].iloc[i] > 0:
            i_ucross12 = i
            num_all_cross += 1
        if df["cross23_u"].iloc[i] > 0:
            i_ucross23 = i
            num_all_cross += 1
        if df["cross34_u"].iloc[i] > 0:
            i_ucross34 = i
            num_all_cross += 1
        if df["cross12_d"].iloc[i] > 0:
            i_dcross12 = i
            num_all_cross += 1
        if df["cross23_d"].iloc[i] > 0:
            i_dcross23 = i
            num_all_cross += 1
        if df["cross34_d"].iloc[i] > 0:
            i_dcross34 = i
            num_all_cross += 1
    
    ref_i, ref_other_i = 10, 15
    ema1, ema2, ema3, ema4 = df["ema1"].iloc[curr_idx], df["ema2"].iloc[curr_idx], df["ema3"].iloc[curr_idx], df["ema4"].iloc[curr_idx]
    avg_candle_length = np.mean(np.abs(np.array(df["open"]) - np.array(df["close"])))
    gap = GAP_CORR*avg_candle_length
    
    def arranged_triangles(i_dcross12, i_dcross23, i_dcross34, i_ucross12, i_ucross23, i_ucross34, ref_other_i):  # for long, opposite arguments for short position
        if 0 < i_ucross12 and limit-ref_i < i_ucross34:
            if i_ucross12-ref_other_i <= np.max([i_dcross34, i_dcross23, i_dcross12]) < i_ucross12 and i_ucross12 <= i_ucross23 and i_ucross23 <= i_ucross34:
                return True
        print("\t| not arranged triangles ")
        
    def zigzag(num_all_cross):
        if num_all_cross > 12:
            return True
        print("\t| not zigzag")
        
        
    if zigzag(num_all_cross):
        if ema1> ema2> ema3> ema4 and arranged_triangles(i_dcross12, i_dcross23, i_dcross34, i_ucross12, i_ucross23, i_ucross34, ref_other_i):
            if ema3 < curr_price:
                ent_price2 = curr_price
                curr_price2 = True
            
            if ent_price2:
                pattern = "Ascending Arrangement - will go down"
                tp_price2 = max(ent_price2 - gap, np.min(df["ema1"]))
                sl_price2 = ent_price2 + gap
               
        if ema1< ema2< ema3< ema4 and arranged_triangles(i_ucross12, i_ucross23, i_ucross34, i_dcross12, i_dcross23, i_dcross34, ref_other_i):
    
            if ema3 > curr_price:
                ent_price1 = curr_price 
                curr_price1 = True
            if ent_price1:
                pattern = "desending Arrangement - will go up"
                tp_price1 = min(ent_price1 + gap, np.max(df["ema1"]))
                sl_price1 = ent_price1 - gap
                
    if plot:
        ax.set_title(f"{pattern} - {sym}, {tf}", position = (0.5,1.05),fontsize = 18)
        
        if ent_price1:
            ax.scatter(df["Index"].iloc[-1], ent_price1, color=BUY_COLOR, alpha = 0.6, s=50, marker="s")
            ax.scatter(df["Index"].iloc[-1], sl_price1, color=BUY_COLOR, alpha = 0.6, s=50, marker="x")
            ax.scatter(df["Index"].iloc[-1], tp_price1, color=BUY_COLOR, alpha = 0.6, s=50, marker="*")
            
        if ent_price2:
            ax.scatter(df["Index"].iloc[-1], ent_price2, color=SELL_COLOR, alpha = 0.6, s=50, marker="s")
            ax.scatter(df["Index"].iloc[-1], sl_price2, color=SELL_COLOR, alpha = 0.6, s=50, marker="x")
            ax.scatter(df["Index"].iloc[-1], tp_price2, color=SELL_COLOR, alpha = 0.6, s=50, marker="*")

        os.makedirs("Figures/", exist_ok=True)
        plt.savefig(f"Figures/{imgfilename}.jpg", dpi = 300)
        plt.close()
    if not pattern:
        return decided_res
    
    return {
            "pattern": pattern,
            "ent_price1": ent_price1,
            "position1": LONG,
            "tp_price1": tp_price1,
            "sl_price1": sl_price1,
            "curr_price1": curr_price1,
            
            "ent_price2": ent_price2,
            "position2": SHORT,
            "tp_price2": tp_price2,
            "sl_price2": sl_price2,
            "curr_price2": curr_price2,
        }
    
    

async def tracking(sym, position, ent_price, sl_price, tp_price, open_to_buy_more, tf = TF, limit = 60, n=5, imgfilename="realtime", decided_res=None):
    binance = get_binance()
    df = await past_data(binance, sym, tf, limit+n)
    df = add_emas(df)
    df = df.iloc[-limit:]
    await binance.close()
    
    curr_idx = len(df["close"])-1
    curr_price = df["close"].iloc[-1]
    df["Index"] = df.index
    
    ema1, ema2, ema3, ema4 = df["ema1"].iloc[curr_idx], df["ema2"].iloc[curr_idx], df["ema3"].iloc[curr_idx], df["ema4"].iloc[curr_idx]
    avg_candle_length = np.mean(np.abs(np.array(df["open"]) - np.array(df["close"])))
    gap = GAP_CORR*(np.abs(ema1 - ema2) + avg_candle_length)
    
    sl_close, tp_close = False, False
    buy_more = False
    if position == LONG:
        # tp_price = max(ema1, ent_price + 2*gap)
        # sl_price = min(ema1, ent_price) - gap
        if curr_price > tp_price:
            tp_close = True
        elif curr_price < sl_price:
            sl_close = True
    else:
        # tp_price = min(ema1, ent_price - 2*gap)
        # sl_price = max(ema1, ent_price) + gap
        if curr_price < tp_price:
            tp_close = True
        elif curr_price > sl_price:
            sl_close = True
            
    ## Plot
    plot=True
    if plot:
        RED, ORANGE, YELLOW, GREEN = (0.6, 0, 0, 1), (0.7, 0.5, 0, 1), (0.6, 0.6, 0, 1), (0.1, 0.5, 0, 1)
        RED2, ORANGE2, YELLOW2 = (0.9, 0.3, 0, 1), (0.9, 0.7, 0, 1), (0.6, 0.9, 0, 1)
        plt.rcParams["figure.figsize"] = (6,4)
        f, ax = plt.subplots(1,1)
        ax.set_facecolor((0.9, 0.9, 0.85))
        plt.subplots_adjust(top = 0.9, bottom = 0.05, right = 0.98, left = 0.1, 
                    hspace = 0.4, wspace = 0.4)
        ax.plot(df["Index"], df["ema1"], color=RED)
        ax.plot(df["Index"], df["ema2"], color=ORANGE)
        ax.plot(df["Index"], df["ema3"], color=YELLOW)
        ax.plot(df["Index"], df["ema4"], color=GREEN)
        ax.plot(df["Index"], df["ema5"], color="k")
        candlestick_ohlc(ax, df.loc[:, ["Index", "open", "high", "low", "close"] ].values, width=0.6, colorup='green', colordown='red', alpha=0.8)
        markersize = 60
        ax.scatter(df["Index"], df["cross12_d"], color=RED2, marker="v", s=markersize, zorder=5)
        ax.scatter(df["Index"], df["cross12_u"], color=RED2, marker="^", s=markersize, zorder=5)
        ax.scatter(df["Index"], df["cross23_d"], color=ORANGE2, marker="v", s=markersize, zorder=5)
        ax.scatter(df["Index"], df["cross23_u"], color=ORANGE2, marker="^", s=markersize, zorder=5)
        ax.scatter(df["Index"], df["cross34_d"], color=YELLOW2, marker="v", s=markersize, zorder=5)
        ax.scatter(df["Index"], df["cross34_u"], color=YELLOW2, marker="^", s=markersize, zorder=5)
        
        ax.set_title(f"{position} - {sym}, {tf}", position = (0.5,1.05),fontsize = 18)
        pcol = BUY_COLOR if position == LONG else SELL_COLOR
        ax.axhline(ent_price, color=pcol)
        ax.axhline(sl_price, color=SELL_COLOR, ls="--")
        ax.axhline(tp_price, color=BUY_COLOR, ls="-.")
            
        os.makedirs("Figures/", exist_ok=True)
        plt.savefig(f"Figures/{imgfilename}.jpg", dpi = 300)
        plt.close()
    
    return tp_price, sl_price, tp_close, sl_close, buy_more