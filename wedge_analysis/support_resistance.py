import os
from wedge_analysis.wedge import *
from mplfinance.original_flavor import candlestick_ohlc
from utils import *
from HYPERPARAMETERS import *
from cal_utils import cal_rsi, cal_srsi
pd.set_option('mode.chained_assignment',  None)

TF = "5m"
WINDOW = 4
LIMIT = 100

def draw_line(point1, point2, ax):
    x_values = [point1[0], point2[0]]
    y_values = [point1[1], point2[1]]
    ax.plot(x_values, y_values, linewidth=5, alpha=0.3, color='black') 


def detect_trend(pivots, threshold: float = 0.7) -> str:
    if len(pivots) < 3:
        return "not enough data"
    increases = 0
    decreases = 0
    total = 0

    for i in range(1, len(pivots)):
        if pivots[i] > pivots[i - 1]:
            increases += 1
        elif pivots[i] < pivots[i - 1]:
            decreases += 1
        total += 1

    up_ratio = increases / total
    down_ratio = decreases / total

    if up_ratio >= threshold:
        return "uptrend"
    elif down_ratio >= threshold:
        return "downtrend"
    else:
        return "sideways"

"""
over and under
+ divergence
"""

async def find_support_resistance(sym, pnl, tf = TF, limit = LIMIT, imgfilename="realtime", decided_res=None):
    df = await past_data(sym, tf, limit)
    avg_candle_length = np.mean(np.abs(np.array(df["open"]) - np.array(df["close"])))
    rsi = cal_rsi(df["close"], n=14)
    df['rsi'] = rsi
    df.reset_index(drop=False, inplace=True)
    
    df["Pivot"] = 0
    df["Pivot"] = df.apply(lambda x: pivot_id(df, x.name, WINDOW, WINDOW), axis=1)
    df.loc[len(df)-1, "Pivot"] = 0
    df['PointPos'] = df.apply(lambda x: pivot_point_position(x), axis=1) # Used for visualising the pivot points
    # df.reset_index(drop=True, inplace=True)
    
    pattern = None
    ent_price1, tp_price1, sl_price1, curr_price1 = None, None, None, False
    ent_price2, tp_price2, sl_price2, curr_price2 = None, None, None, False
    curr_idx = len(df["close"])-1
    curr_price = df["close"].iloc[curr_idx]
    df["Index"] = df.index
    
    support_idx = np.where(df["Pivot"] == 1)[0]  # low
    s_ps_df = df['PointPos'].loc[support_idx]
    resist_idx = np.where(df["Pivot"] == 2)[0]   # high
    r_ps_df = df['PointPos'].loc[resist_idx]
    
    s_ps = np.array(s_ps_df)
    r_ps = np.array(r_ps_df)
    
    if len(s_ps) < 4 or len(r_ps) < 4:
        if avg_candle_length > curr_price*0.0001:
            if curr_price > df["open"].iloc[0] + 5*avg_candle_length:
                pattern = "Scalping - long"
                ent_price1 = curr_price
                curr_price1 = True
                tp_price1 = curr_price + max(curr_price*0.005, 10*avg_candle_length)
                sl_price1 = curr_price - (tp_price1 - curr_price)*0.5
                
            if curr_price < df["open"].iloc[0] - 5*avg_candle_length:
                pattern = "Scalping - short"
                ent_price2 = curr_price
                curr_price2 = True
                tp_price2 = curr_price - max(curr_price*0.005, 10*avg_candle_length)
                sl_price2 = curr_price + (curr_price - tp_price2)*0.5
        else:
            return None
    else:
        # if (s_ps[-1] > s_ps[-2] > s_ps[-3] and r_ps[-1] > r_ps[-2] > r_ps[-3]) or\
        if s_ps[-1] > s_ps[-2] > s_ps[-3] > s_ps[-4]:
            if resist_idx[-1] < support_idx[-1]:
                pattern = "Scalping - long"
                ent_price1 = curr_price
                curr_price1 = True
                tp_price1 = curr_price + max(curr_price*0.005, 10*avg_candle_length)
                sl_price1 = curr_price - (tp_price1 - curr_price)*0.5
    
        # if (r_ps[-1] < r_ps[-2] < r_ps[-3] and s_ps[-1] < s_ps[-2] < s_ps[-3]) or\
        if r_ps[-1] < r_ps[-2] < r_ps[-3] < r_ps[-4]:
            if support_idx[-1] < resist_idx[-1]:
                pattern = "Scalping - short"
                ent_price2 = curr_price
                curr_price2 = True
                tp_price2 = curr_price - max(curr_price*0.005, 10*avg_candle_length)
                sl_price2 = curr_price + (curr_price - tp_price2)*0.5
        
    ## Plot
    plt.rcParams["figure.figsize"] = (6,6)
    f, (ax, ax_rsi) = plt.subplots(2,1, gridspec_kw={'height_ratios': [3, 1]})
    ax.set_facecolor((0.95, 0.95, 0.9))
    plt.subplots_adjust(top = 0.9, bottom = 0.05, right = 0.98, left = 0.1, 
                hspace = 0.4, wspace = 0.4)
    candlestick_ohlc(ax, df.loc[:, ["Index", "open", "high", "low", "close"] ].values, width=0.6, colorup='green', colordown='red', alpha=0.8)
    ax.scatter(df["Index"], df["PointPos"], color="b", s=20)
    ax.set_title(f"{pattern} - {sym}, {tf}", position = (0.5,1.05),fontsize = 18)
    
    ax_rsi.plot(df["rsi"], color=(0.2, 0.3, 0.8))
    ax_rsi.set_title('RSI',position = (0.5,1.05),fontsize = 15)
    ax_rsi.axhline(30, ls = '--', c='y', alpha = 0.9)
    ax_rsi.axhline(70, ls = '--', c='y', alpha = 0.9)
    
    if ent_price1:
        ax.scatter(curr_idx, ent_price1, color=BUY_COLOR, alpha = 0.6, s=50, marker="s")
        ax.scatter(curr_idx, sl_price1, color=BUY_COLOR, alpha = 0.6, s=50, marker="x")
        ax.scatter(curr_idx, tp_price1, color=BUY_COLOR, alpha = 0.6, s=50, marker="*")
        
    if ent_price2:
        ax.scatter(curr_idx, ent_price2, color=SELL_COLOR, alpha = 0.6, s=50, marker="s")
        ax.scatter(curr_idx, sl_price2, color=SELL_COLOR, alpha = 0.6, s=50, marker="x")
        ax.scatter(curr_idx, tp_price2, color=SELL_COLOR, alpha = 0.6, s=50, marker="*")

    os.makedirs("Figures/", exist_ok=True)
    plt.savefig(f"Figures/{imgfilename}.jpg", dpi = 300)
    plt.close()
        
    if not pattern:
        return decided_res
    
    now = datetime.now()
    now = now.strftime("%m%d_%H%M%S")
    os.rename(f"Figures/{imgfilename}.jpg", f"logs2/{now}_{sym.split('/')[0]}.jpg")
    
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
    
    

async def tracking(sym, position, ent_price, sl_price, tp_price, open_to_buy_more, tf = TF, limit = LIMIT, n=5, imgfilename="realtime", decided_res=None):
    df = await past_data(sym, tf, limit+n)
    df = df.iloc[-limit:]
    rsi = cal_rsi(df["close"], n=14)
    df['rsi'] = rsi
    df.reset_index(drop=False, inplace=True)
    
    df["Pivot"] = 0
    df["Pivot"] = df.apply(lambda x: pivot_id(df, x.name, WINDOW, WINDOW), axis=1)
    df.loc[len(df)-1, "Pivot"] = 0
    df['PointPos'] = df.apply(lambda x: pivot_point_position(x), axis=1) # Used for visualising the pivot points
    df.reset_index(drop=True, inplace=True)
    
    curr_idx = len(df["close"])-1
    curr_price = df["close"].iloc[curr_idx]
    df["Index"] = df.index
    
    sl_close, tp_close = False, False
    buy_more = False
    if position == LONG:
        if curr_price > tp_price:
            tp_close = True
        elif curr_price < sl_price:
            sl_close = True
    else:
        if curr_price < tp_price:
            tp_close = True
        elif curr_price > sl_price:
            sl_close = True

    ## Plot
    plt.rcParams["figure.figsize"] = (6,6)
    f, (ax, ax_rsi) = plt.subplots(2,1, gridspec_kw={'height_ratios': [3, 1]})
    ax.set_facecolor((0.95, 0.95, 0.9))
    plt.subplots_adjust(top = 0.9, bottom = 0.05, right = 0.98, left = 0.1, 
                hspace = 0.4, wspace = 0.4)
    candlestick_ohlc(ax, df.loc[:, ["Index", "open", "high", "low", "close"] ].values, width=0.6, colorup='green', colordown='red', alpha=0.8)
    ax.scatter(df["Index"], df["PointPos"], color="b", s=20)
    ax.set_title(f"{position} - {sym}, {tf}", position = (0.5,1.05), fontsize = 18)

    ax_rsi.plot(df["rsi"], color=(0.2, 0.3, 0.8))
    ax_rsi.set_title('RSI',position = (0.5,1.05), fontsize = 15)
    ax_rsi.axhline(25, ls = '--', c='y', alpha = 0.9)
    ax_rsi.axhline(75, ls = '--', c='y', alpha = 0.9)
    
    pcol = BUY_COLOR if position == LONG else SELL_COLOR
    
    ax.axhline(ent_price, color=pcol)
    ax.axhline(sl_price, color=SELL_COLOR, ls="--")
    ax.axhline(tp_price, color=BUY_COLOR, ls="-.")
    
    os.makedirs("Figures/", exist_ok=True)
    plt.savefig(f"Figures/{imgfilename}.jpg", dpi = 300)
    plt.close()
    
    return tp_price, sl_price, tp_close, sl_close, buy_more