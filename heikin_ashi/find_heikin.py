from wedge_analysis.trendline_automation import *
from wedge_analysis.wedge import *
from utils import *
from cal_utils import cal_srsi
from ccxt.base.errors import BadSymbol
pd.set_option('mode.chained_assignment',  None)


def whether_heikin_candle(df):
    # close: pd.Series
    for i in range(2,4):
        this, before = df.iloc[-i], df.iloc[-i-1]
        this_body = this["close"] - this["open"]
        this_len = np.abs(this_body)
        before_body = before["close"] - before["open"]
        before_len = np.abs(before_body)
        if not (this_len > before_len and this_body*before_body > 0):
            return False
    if this_body > 0:
        return LONG
    else:
        return SHORT
            
def whether_near_to_ema200(df):
    this= df.iloc[-1]
    this_body = this["close"] - this["open"]
    if this_body > 0:
        diff = (df["low"] - df["ema200"])[-15:]
        amin = np.argmin(np.abs(diff))
        if amin in [13, 14, 15]:
            return True
    if this_body < 0:
        diff = (df["high"] - df["ema200"])[-15:]
        amin = np.argmin(np.abs(diff))
        if amin in [12, 13, 14, 15]:
            return True
    return False
            
            
async def find_heikin(sym, tf = "5m", limit = 60, imgfilename="realtime"):
    binance = get_binance()
    df = await past_data(binance, sym, tf, limit+200)
    df['ema200'] = df['close'].ewm(50).mean()
    
    n = 5
    df = df.iloc[-limit-n:]
    srsi_k, srsi_d = cal_srsi(df["close"], n=n)
    df['srsi_k'] = srsi_k
    df['srsi_d'] = srsi_d
    
    df = df.iloc[-limit:]
    df.reset_index(drop=True, inplace=True)
    await binance.close()
    
    # long hammer?
    # position = whether_heikin_candle(df)
    # near to EMA200?
    # near_to_ema200 = whether_near_to_ema200(df)
    
    pattern = None
    ent_price1 = None
    close_price1 = None
    ent_price2 = None
    close_price2 = None
    curr_idx = len(df["close"])-1
    curr_price = df["close"].iloc[-1]
    
    if df['ema200'].iloc[-1] < df['ema200'].iloc[-5] and\
        min(df["srsi_d"].iloc[-2:]) > 0 and min(df["srsi_d"].iloc[-2:]) < 25 and\
        df["srsi_k"].iloc[-1] > df["srsi_d"].iloc[-1] and df["srsi_k"].iloc[-3] < df["srsi_d"].iloc[-3] :
        pattern = "HeikinAshi - SHOrt"
        ent_price2 = curr_price
        close_price2 = -0.006*curr_price + curr_price
        
    if df['ema200'].iloc[-1] > df['ema200'].iloc[-5] and\
        max(df["srsi_d"].iloc[-2:]) < 100 and max(df["srsi_d"].iloc[-2:]) > 75 and \
        df["srsi_k"].iloc[-1] < df["srsi_d"].iloc[-1] and df["srsi_k"].iloc[-3] > df["srsi_d"].iloc[-3]:
        pattern = "HeikinAshi - LONG"
        ent_price1 = curr_price
        close_price1 = 0.006*curr_price + curr_price
    
    ## Plot
    plt.rcParams["figure.figsize"] = (6,6)
    plt.subplots_adjust(top = 0.95, bottom = 0.05, right = 0.98, left = 0.05, 
                hspace = 1, wspace = 1)
    f, (ax, ax_rsi) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]})
    ax.set_facecolor((0.9, 0.9, 0.85))
    
    df["Index"] = df.index
    candlestick_ohlc(ax, df.loc[:, ["Index", "open", "high", "low", "close"] ].values, width=0.6, colorup='green', colordown='red', alpha=0.8)
    ax.set_title(f"Heikin - {sym}, {tf}", position = (0.5,1.05),fontsize = 23)
    ax.plot(df["ema200"], color=(0.7, 0.7, 0.9))
    ax_rsi.plot(df["srsi_k"], color=(0.2, 0.3, 0.8))
    ax_rsi.plot(df["srsi_d"], color=(0.9, 0.6, 0.1))
    ax_rsi.set_title('S-RSI',position = (0.5,1.05),fontsize = 23)
    ax_rsi.axhline(25, ls = '--', c='y', alpha = 0.9)
    ax_rsi.axhline(75, ls = '--', c='y', alpha = 0.9)

        
    if not pattern:
        print(f"Heikin- No pattern")
        plt.savefig(f"Figures/{imgfilename}.jpg", dpi = 300)
        plt.close()
        return None
        
    if ent_price1:
        ax.scatter(curr_idx, ent_price1, c='g', alpha = 0.9, s=100, marker="s")
        ax.scatter(curr_idx, close_price1, c='g', alpha = 0.9, s=100, marker="x")
    if ent_price2:
        ax.scatter(curr_idx, ent_price2, c='r', alpha = 0.9, s=100, marker="s")
        ax.scatter(curr_idx, close_price2, c='r', alpha = 0.9, s=100, marker="x")


    os.makedirs("Figures/", exist_ok=True)
    plt.savefig(f"Figures/{imgfilename}.jpg", dpi = 300)
    plt.close()
    
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
    