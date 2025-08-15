import os
from wedge_analysis.wedge import *
from mplfinance.original_flavor import candlestick_ohlc
from utils import *
from HYPERPARAMETERS import *
from cal_utils import cal_rsi, cal_srsi
pd.set_option('mode.chained_assignment',  None)

TF1, TF2 = "15m", "1h"
WINDOW = 4
LIMIT = 70



def is_www_pattern(df, lookback=30):
    if len(df) < lookback:
        return False

    recent_data = df.tail(lookback)
    
    bb_width = (recent_data["bb_upper2"] - recent_data["bb_lower2"]) / recent_data["bb_middle"]
    width_std = bb_width.std()
    width_mean = bb_width.mean()
    width_cv = width_std / width_mean 
    
    # print(f"\nBB Width Mean: {width_mean:.4f}, Std: {width_std:.4f}, CV: {width_cv:.4f}")
    
    width_stable = width_cv < 0.4
    
    bb_upper = recent_data["bb_upper2"]
    bb_lower = recent_data["bb_lower2"]
    bb_middle = recent_data["bb_middle"]
    
    x = range(len(bb_upper))
    
    upper_slope = np.polyfit(x, bb_upper, 1)[0]
    upper_slope_ratio = abs(upper_slope) / bb_upper.mean()
    
    lower_slope = np.polyfit(x, bb_lower, 1)[0]
    lower_slope_ratio = abs(lower_slope) / bb_lower.mean()
    
    middle_slope = np.polyfit(x, bb_middle, 1)[0]
    middle_slope_ratio = abs(middle_slope) / bb_middle.mean()
    
    # print(f"Slopes - Upper: {upper_slope_ratio:.6f}, Lower: {lower_slope_ratio:.6f}, Middle: {middle_slope_ratio:.6f}")
    
    bands_flat = (upper_slope_ratio < 0.002 and 
                lower_slope_ratio < 0.002 and 
                middle_slope_ratio < 0.002)
    
    slope_diff = abs(upper_slope - lower_slope)
    slope_diff_ratio = slope_diff / max(abs(upper_slope), abs(lower_slope), 1e-10)
    
    # print(f"Slope Difference Ratio: {slope_diff_ratio:.6f}")
    
    bands_parallel = slope_diff_ratio < 0.9
    
    # 최종 판단
    is_sideways = width_stable and bands_flat and bands_parallel
    
    # print(f"Width Stable: {width_stable}, Bands Flat: {bands_flat}, Bands Parallel: {bands_parallel}")
    # print(f"Final WWW Pattern: {is_sideways}")
    
    return is_sideways


def determine_trend(df2):
    long_ema_5 = df2["ema_5"].iloc[-1]
    long_ema_10 = df2["ema_10"].iloc[-1]
    long_uptrend = long_ema_5 > long_ema_10
    long_downtrend = long_ema_5 < long_ema_10
    
    if long_uptrend:
        return "uptrend"
    elif long_downtrend:
        return "downtrend"
    else:
        return "sideways"


async def find_support_resistance(sym, pnl, tf = TF1, tf2=TF2, limit = LIMIT, imgfilename="realtime", decided_res=None):
    df = await past_data(sym, tf, limit)
    df2 = await past_data(sym, tf2, limit)
    df.reset_index(drop=True, inplace=True)
    df2.reset_index(drop=True, inplace=True)

    close = df["close"]
    close2 = df2["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    
    df["range"] = df["high"] - df["low"]
    avg_range = df["range"].mean()
    
    df["rsi"] = cal_rsi(close, n=min(14, len(df)-1))
    df["ema_5"] = close.ewm(span=5, adjust=False).mean()
    df["ema_10"] = close.ewm(span=10, adjust=False).mean()
    df["ema_20"] = close.ewm(span=20, adjust=False).mean()
    df2["ema_5"] = close2.ewm(span=5, adjust=False).mean()
    df2["ema_10"] = close2.ewm(span=10, adjust=False).mean()
    df2["ema_20"] = close2.ewm(span=20, adjust=False).mean()
    
    df["vol_avg"] = volume.rolling(10).mean()
    df["volume_ma"] = volume.rolling(20).mean()
    df["volume_ratio"] = volume / df["volume_ma"]
    df["volume_ratio"] = df["volume_ratio"].fillna(1.0)  # NaN 값을 1.0으로 채우기
    
    period = 20
    reward_ratio = 1.5
    ref = 1.5
    
    df["bb_middle"] = close.rolling(period).mean()
    df["bb_std"] = close.rolling(period).std()
    df["bb_upper1"] = df["bb_middle"] + (df["bb_std"] * 2)  # 표준편차 3
    df["bb_lower1"] = df["bb_middle"] - (df["bb_std"] * 2)
    df["bb_upper2"] = df["bb_middle"] + (df["bb_std"] * 3)  # 표준편차 3
    df["bb_lower2"] = df["bb_middle"] - (df["bb_std"] * 3)

    df["vwap"] = (close * volume).cumsum() / volume.cumsum()

    curr_price = close.iloc[-1]
    curr_rsi = df["rsi"].iloc[-1]
    curr_bb_upper = df["bb_upper1"].iloc[-1]
    curr_bb_lower = df["bb_lower1"].iloc[-1]
    curr_bb_middle = df["bb_middle"].iloc[-1]

    pattern = None
    ent_price1, tp_price1, sl_price1, curr_price1 = None, None, None, False
    ent_price2, tp_price2, sl_price2, curr_price2 = None, None, None, False

    www = is_www_pattern(df)

    market_trend = determine_trend(df2)

    bb_touch_lower = curr_price <= curr_bb_lower  # 하단 터치
    bb_touch_upper = curr_price >= curr_bb_upper  # 상단 터치

    if www:
        if market_trend == "uptrend":
            if bb_touch_lower:
                ent_price1 = curr_price
                # sl_price1 = curr_price - avg_range * ref
                sl_price1 = df["bb_lower2"].iloc[-1] - avg_range * ref
                tp_price1 = max(curr_bb_middle, curr_price + (curr_price - sl_price1) * reward_ratio)
                curr_price1 = True
                pattern = "WWW Uptrend Long"
                
        elif market_trend == "downtrend":
            if bb_touch_upper:
                ent_price2 = curr_price
                # sl_price2 = curr_price + avg_range * ref
                sl_price2 = df["bb_upper2"].iloc[-1] + avg_range * ref
                tp_price2 = min(curr_bb_middle, curr_price - (sl_price2 - curr_price) * reward_ratio)
                curr_price2 = True
                pattern = "WWW Downtrend Short"
                
    if 1:
        import matplotlib.pyplot as plt
        RED, ORANGE, YELLOW, GREEN = (0.6, 0, 0, 1), (0.7, 0.5, 0, 1), (0.6, 0.6, 0, 1), (0.1, 0.5, 0, 1)
        BLUE, PURPLE = (0.2, 0.3, 0.8), (0.5, 0.1, 0.6)
        plt.rcParams["figure.figsize"] = (10, 10)
        df["Index"] = df.index
        df2["Index"] = df2.index
        curr_idx = len(df)
        f, (ax, ax2, ax_vol, ax_rsi) = plt.subplots(4, 1, gridspec_kw={'height_ratios': [3, 3, 1, 1]})
        ax.set_facecolor((0.95, 0.95, 0.9))
        ax2.set_facecolor((0.95, 0.95, 0.9))
        plt.subplots_adjust(top=0.9, bottom=0.05, right=0.98, left=0.1, hspace=0.5)

        candlestick_ohlc(ax, df.loc[:, ["Index", "open", "high", "low", "close"]].values, width=0.6, colorup='green', colordown='red', alpha=0.8)
        candlestick_ohlc(ax2, df2.loc[:, ["Index", "open", "high", "low", "close"]].values, width=0.6, colorup='green', colordown='red', alpha=0.8)
        
        title = f"{pattern} - {sym} {tf} {market_trend} www:{www}"
        ax.set_title(title, position=(0.5, 1.05), fontsize=14)
        ax2.set_title(market_trend, position=(0.5, 1.05), fontsize=14)
        
        # 핵심 지표만 표시
        ax.plot(df["Index"], df["vwap"], color=PURPLE, linestyle='--', label='VWAP', linewidth=2)
        ax.plot(df["Index"], df["bb_upper1"], color=YELLOW, linestyle='--', label='BB2', alpha=0.8)
        ax.plot(df["Index"], df["bb_lower1"], color=YELLOW, linestyle='--', alpha=0.8)
        ax.plot(df["Index"], df["bb_upper2"], color=ORANGE, linestyle='--', label='BB3', alpha=0.8)
        ax.plot(df["Index"], df["bb_lower2"], color=ORANGE, linestyle='--', alpha=0.8)
        ax.legend(loc='upper left', fontsize=10)
        
        # 볼륨 (볼륨 비율 색상 코딩)
        colors = ['darkred' if ratio > 1.5 else 'red' if ratio > 1.2 else 'orange' if ratio > 1.0 else 'gray' 
                 for ratio in df["volume_ratio"]]
        ax_vol.bar(df["Index"], df["volume"], color=colors, alpha=0.7)
        ax_vol.axhline(df["volume_ma"].iloc[-1], color='blue', linestyle='--', alpha=0.7)

        # RSI 차트
        ax_rsi.plot(df["Index"], df["rsi"], color=BLUE, label='RSI(14)', linewidth=2)
        ax_rsi.axhline(30, ls='--', c='g', alpha=0.7)
        ax_rsi.axhline(70, ls='--', c='r', alpha=0.7)
        ax_rsi.axhline(25, ls=':', c='g', alpha=0.5)
        ax_rsi.axhline(75, ls=':', c='r', alpha=0.5)
        ax_rsi.set_title(f'RSI (Current: {curr_rsi:.1f})', fontsize=12)
        ax_rsi.legend(fontsize=10)
        
        # 진입/손절/익절 포인트
        if ent_price1:
            ax.scatter(curr_idx, ent_price1, color='limegreen', s=100, marker="^", label="LONG", zorder=5)
            ax.axhline(sl_price1, color='red', linestyle=':', alpha=0.8, linewidth=2)
            ax.axhline(tp_price1, color='green', linestyle=':', alpha=0.8, linewidth=2)
            
        if ent_price2:
            ax.scatter(curr_idx, ent_price2, color='red', s=100, marker="v", label="SHORT", zorder=5)
            ax.axhline(sl_price2, color='red', linestyle=':', alpha=0.8, linewidth=2)
            ax.axhline(tp_price2, color='green', linestyle=':', alpha=0.8, linewidth=2)

    os.makedirs("Figures/", exist_ok=True)
    plt.savefig(f"Figures/{imgfilename}.jpg", dpi = 100)
    plt.close()
        
    if not pattern:
        return decided_res
    
    # from datetime import datetime
    # now = datetime.now()
    # now = now.strftime("%m%d_%H%M%S")
    # os.rename(f"Figures/{imgfilename}.jpg", f"logs2/{now}_{sym.split('/')[0]}.jpg")
    
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
    
    

async def tracking(sym, position, ent_price, sl_price, tp_price, open_to_buy_more, tf = TF1, limit = LIMIT, n=5, imgfilename="realtime", decided_res=None):
    df = await past_data(sym, tf, limit+n)
    df = df.iloc[-limit:]
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    
    # 정상 계산
    df["rsi"] = cal_rsi(close, n=min(14, len(df)-1))
    df["ema_5"] = close.ewm(span=min(5, len(df)//2), adjust=False).mean()
    df["ema_10"] = close.ewm(span=min(10, len(df)//2), adjust=False).mean()
    df["ema_20"] = close.ewm(span=min(20, len(df)-1), adjust=False).mean()
    df["vol_avg"] = volume.rolling(min(10, len(df)//2)).mean()
    
    # 볼륨 관련 지표 추가
    df["volume_ma"] = volume.rolling(min(20, len(df)-1)).mean()
    df["volume_ratio"] = volume / df["volume_ma"]
    df["volume_ratio"] = df["volume_ratio"].fillna(1.0)  # NaN 값을 1.0으로 채우기
    
    period = 30
    
    df["bb_middle"] = close.rolling(period).mean()
    df["bb_std"] = close.rolling(period).std()
    df["bb_upper1"] = df["bb_middle"] + (df["bb_std"] * 2)  # 표준편차 3
    df["bb_lower1"] = df["bb_middle"] - (df["bb_std"] * 2)
    df["bb_upper2"] = df["bb_middle"] + (df["bb_std"] * 3)  # 표준편차 3
    df["bb_lower2"] = df["bb_middle"] - (df["bb_std"] * 3)

    # VWAP
    df["vwap"] = (close * volume).cumsum() / volume.cumsum()
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
    import matplotlib.pyplot as plt
    RED, ORANGE, YELLOW, GREEN = (0.6, 0, 0, 1), (0.7, 0.5, 0, 1), (0.6, 0.6, 0, 1), (0.1, 0.5, 0, 1)
    BLUE, PURPLE = (0.2, 0.3, 0.8), (0.5, 0.1, 0.6)
    plt.rcParams["figure.figsize"] = (10, 10)
    f, (ax, ax_vol, ax_rsi) = plt.subplots(3, 1, gridspec_kw={'height_ratios': [3, 1, 1]})
    ax.set_facecolor((0.95, 0.95, 0.9))
    plt.subplots_adjust(top=0.9, bottom=0.05, right=0.98, left=0.1, hspace=0.5)

    candlestick_ohlc(ax, df.loc[:, ["Index", "open", "high", "low", "close"]].values, width=0.6, colorup='green', colordown='red', alpha=0.8)
    
    title = f"{position} - {sym} {tf}"
    ax.set_title(title, position=(0.5, 1.05), fontsize=14)
    
    # 핵심 지표만 표시
    ax.plot(df["Index"], df["vwap"], color=PURPLE, linestyle='--', label='VWAP', linewidth=2)
    ax.plot(df["Index"], df["bb_upper1"], color=YELLOW, linestyle='--', label='BB2', alpha=0.8)
    ax.plot(df["Index"], df["bb_lower1"], color=YELLOW, linestyle='--', alpha=0.8)
    ax.plot(df["Index"], df["bb_upper2"], color=ORANGE, linestyle='--', label='BB3', alpha=0.8)
    ax.plot(df["Index"], df["bb_lower2"], color=ORANGE, linestyle='--', alpha=0.8)
    ax.legend(loc='upper left', fontsize=10)
    
    # 볼륨 (볼륨 비율 색상 코딩)
    colors = ['darkred' if ratio > 1.5 else 'red' if ratio > 1.2 else 'orange' if ratio > 1.0 else 'gray' 
                for ratio in df["volume_ratio"]]
    ax_vol.bar(df["Index"], df["volume"], color=colors, alpha=0.7)
    ax_vol.axhline(df["volume_ma"].iloc[-1], color='blue', linestyle='--', alpha=0.7)

    # RSI 차트
    curr_rsi = df["rsi"].iloc[-1] if not pd.isna(df["rsi"].iloc[-1]) else 50  # RSI가 NaN인 경우 기본값
    ax_rsi.plot(df["Index"], df["rsi"], color=BLUE, label='RSI(14)', linewidth=2)
    ax_rsi.axhline(30, ls='--', c='g', alpha=0.7)
    ax_rsi.axhline(70, ls='--', c='r', alpha=0.7)
    ax_rsi.axhline(25, ls=':', c='g', alpha=0.5)
    ax_rsi.axhline(75, ls=':', c='r', alpha=0.5)
    ax_rsi.set_title(f'RSI (Current: {curr_rsi:.1f})', fontsize=12)
    ax_rsi.legend(fontsize=10)
    
    pcol = BUY_COLOR if position == LONG else SELL_COLOR
    
    ax.axhline(ent_price, color=pcol)
    ax.axhline(sl_price, color=SELL_COLOR, ls="--")
    ax.axhline(tp_price, color=BUY_COLOR, ls="-.")
    
    os.makedirs("Figures/", exist_ok=True)
    plt.savefig(f"Figures/{imgfilename}.jpg", dpi = 100)
    plt.close()
    
    return tp_price, sl_price, tp_close, sl_close, buy_more