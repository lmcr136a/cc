import os
import numpy as np
import pandas as pd
import time
import random
import matplotlib.pyplot as plt
import ccxt 
from colorama import Fore, Style, init
from datetime import datetime
from HYPERPARAMETERS import *
import asyncio
import ccxt.pro as ccxtpro
from wedge_analysis.wedge import *
from mplfinance.original_flavor import candlestick_ohlc
from utils import *
from HYPERPARAMETERS import *
from ccxt.base.errors import BadSymbol
from cal_utils import cal_rsi, cal_srsi
from scipy.signal import argrelextrema
from sklearn.cluster import DBSCAN
    
    
ONCE = True

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

async def find_support_resistance_window(df, df2, sym, ref, period=15, reward_ratio=1.5, tf="3m", imgfilename="realtime", window_idx=0):
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
    
    
    # 볼린저 밴드
    # period = min(15, len(df)-1)
    df["bb_middle"] = close.rolling(period).mean()
    df["bb_std"] = close.rolling(period).std()
    df["bb_upper1"] = df["bb_middle"] + (df["bb_std"] * 2)  # 표준편차 3
    df["bb_lower1"] = df["bb_middle"] - (df["bb_std"] * 2)
    df["bb_upper2"] = df["bb_middle"] + (df["bb_std"] * 3)  # 표준편차 3
    df["bb_lower2"] = df["bb_middle"] - (df["bb_std"] * 3)

    # VWAP
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
                
        
    if not pattern:
        return None
    
    # 차트 플롯 및 저장
    if len(os.listdir("sliding_backtest/")) <= 15:
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

        if not pattern:
            pattern = "NONE"
        os.makedirs("sliding_backtest/", exist_ok=True)
        now = datetime.now().strftime("%m%d_%H%M%S")
        plt.savefig(f"sliding_backtest/{now}_{imgfilename}_w{window_idx:04d}_{pattern.replace('(', '').replace(')', '')}.jpg", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        if len(os.listdir("sliding_backtest/")) > 20:
            exit()
        
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


async def run_sliding_backtest(tf, T_window, T_start=100, T_end=500, T_step=5, coef_near_bb=0.002, coef_sl_bb=0.005):
    
    pnls = 100
    running_times = []
    trade_count = 0
    results = {}
    coef_test_results = {}
    signal_count = 0
    cum_profit = 0
    
    large_vol_symlist = []
    
    # binance = get_binance()
    # await binance.load_markets()
    # market = binance.markets
    # symlist = []
    # for s in market.keys():
    #     if s.split(":")[0][-4:] == "USDT":
    #         symlist.append(s.split(":")[0])
    # symlist = list(set(symlist))
    # # symlist = SYMLIST
    # await binance.close()
    # SYMLIST = symlist
    SYMLIST = ["ETH/USDT", "XRP/USDT", "SOL/USDT", "BNB/USDT", "DOGE/USDT", "ADA/USDT", "TRX/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT", "BCH/USDT", "NEAR/USDT", "LTC/USDT", "UNI/USDT", "ICP/USDT", "XLM/USDT"]
    
    # SYMLIST = ["BTC/USDT", "XRP/USDT", "SOL/USDT", "ETH/USDT"]
    random.shuffle(SYMLIST)
    for i, sym in enumerate(SYMLIST):
        try:
            print(f"\n=== Testing {sym} ===")
            
            binance = get_binance()
            try:
                vol = await binance.fetch_tickers(symbols=[sym])
                await binance.close()
                
                if (not len(list(vol.values())) > 0) or list(vol.values())[0]['quoteVolume'] < 1*(10**6):
                    print(f"Skipping {sym} - Low volume")
                    continue
                    
            except BadSymbol:
                await binance.close()
                print(f"Skipping {sym} - Bad symbol")
                continue
            
            large_vol_symlist.append(sym)
            future_t = 100
            total_limit = T_end + future_t  # 여유분 추가
            df_full = await past_data(sym, tf, total_limit)
            df2_full = await past_data(sym, "1h", total_limit)
            print(f"Full data loaded: {len(df_full)} candles")
            
            timeframe_ratio = 4  # 1시간 = 12 * 5분
            # timeframe_ratio = 12  # 1시간 = 12 * 5분
            
            hourly_indices = []
            for idx in range(len(df_full)):
                if df_full.index[idx].minute == 0:  # 정각 데이터
                    hourly_indices.append(idx)
            
            for window_idx, t in enumerate(hourly_indices):
                if t < T_start or t >= len(df_full) - future_t:
                    continue
                    
                df_window = df_full.iloc[t-T_window:t].copy()
                
                current_1h_idx = len(df2_full) - 1 - ((len(df_full) - 1 - t) // timeframe_ratio)
                
                T_window_1h = T_window // timeframe_ratio
                
                end_1h = min(len(df2_full), current_1h_idx + 1)
                
                df2_window = df2_full.iloc[end_1h-T_window-2:end_1h-1].copy()
                current_time = df_window.index[-1]
                current_time_1h = df2_window.index[-1]

                for (period, reward_ratio, ref) in [(20, 1.5, 1.5)]:
                    k = f"{period}_{reward_ratio}_{ref}"
        
                    # 시그널 감지
                    res = await find_support_resistance_window(
                        df_window, df2_window,sym=sym, ref=ref,
                        period=period, reward_ratio=reward_ratio,
                        tf=tf, 
                        imgfilename=f"{tf}", 
                        window_idx=window_idx
                    )
                    
                    if res:
                        signal_count += 1
                        print(f"\n🔍 SIGNAL DETECTED - Window {window_idx} (t={t})")
                        print(f"Pattern: {res['pattern']}")
                        
                        # 진입 포지션 결정
                        if res["ent_price1"] and res["curr_price1"]:
                            ent_price, sl_price, tp_price, position = res["ent_price1"], res["sl_price1"], res["tp_price1"], LONG
                            print(f"LONG - Entry: {ent_price:.6f}, SL: {sl_price:.6f}, TP: {tp_price:.6f}")
                        elif res["ent_price2"] and res["curr_price2"]:
                            ent_price, sl_price, tp_price, position = res["ent_price2"], res["sl_price2"], res["tp_price2"], SHORT
                            print(f"SHORT - Entry: {ent_price:.6f}, SL: {sl_price:.6f}, TP: {tp_price:.6f}")
                        else:
                            print("❌ No valid entry price found")
                            continue
                        
                        # 미래 데이터로 결과 확인 (t 이후 데이터)
                        if t + future_t < len(df_full):  # 충분한 미래 데이터가 있는 경우만
                            future_df = df_full.iloc[t:t+future_t].copy()  # 50개 캔들 미래
                            future_df.reset_index(drop=True, inplace=True)
                            
                            closes = np.array(future_df["close"])
                            tp_close, sl_close, pnl = False, False, 0
                            hit_index = None
                            
                            # 포지션별 손익 계산
                            if position == LONG:
                                tp_is = np.where(closes >= tp_price)[0]
                                sl_is = np.where(closes <= sl_price)[0]
                                if len(tp_is) > 0 and (len(sl_is) == 0 or np.min(tp_is) < np.min(sl_is)):
                                    pnl = (tp_price - ent_price) / ent_price * 100
                                    tp_close = True
                                    hit_index = np.min(tp_is)
                                    print(f"✅ TP HIT at index {hit_index} - PnL: +{pnl:.2f}%")
                                elif len(sl_is) > 0:
                                    pnl = (sl_price - ent_price) / ent_price * 100
                                    sl_close = True
                                    hit_index = np.min(sl_is)
                                    print(f"❌ SL HIT at index {hit_index} - PnL: {pnl:.2f}%")
                                else:
                                    pnl =  0.02
                                    hit_index = len(future_df)
                                    print(f"⏸️ NO CONCLUSION - Current PnL: {pnl:.2f}%")
                                    
                            elif position == SHORT:
                                tp_is = np.where(closes <= tp_price)[0]
                                sl_is = np.where(closes >= sl_price)[0]
                                if len(tp_is) > 0 and (len(sl_is) == 0 or np.min(tp_is) < np.min(sl_is)):
                                    pnl = -(tp_price - ent_price) / ent_price * 100
                                    tp_close = True
                                    hit_index = np.min(tp_is)
                                    print(f"✅ TP HIT at index {hit_index} - PnL: +{pnl:.2f}%")
                                elif len(sl_is) > 0:
                                    pnl = -(sl_price - ent_price) / ent_price * 100
                                    sl_close = True
                                    hit_index = np.min(sl_is)
                                    print(f"❌ SL HIT at index {hit_index} - PnL: {pnl:.2f}%")
                                else:
                                    pnl = 0.02
                                    hit_index = len(future_df)
                                    print(f"⏸️ NO CONCLUSION - Current PnL: {pnl:.2f}%")
                            
                            # 차트 그리기 및 저장
                            if 0:
                                plt.rcParams["figure.figsize"] = (14, 10)
                                f, (ax_main, ax_vol) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]})
                                ax_main.set_facecolor((0.95, 0.95, 0.9))
                                plt.subplots_adjust(top=0.92, bottom=0.08, right=0.98, left=0.08, hspace=0.3)
                                
                                # 메인 차트 (윈도우 + 미래 데이터)
                                plot_data = pd.concat([df_window.iloc[-30:], future_df.iloc[:20]]).reset_index(drop=True)
                                plot_data["Index"] = plot_data.index
                                
                                # 캔들스틱
                                candlestick_ohlc(ax_main, plot_data.loc[:, ["Index", "open", "high", "low", "close"]].values, 
                                            width=0.6, colorup='green', colordown='red', alpha=0.8)
                                
                                # 진입/손절/익절 라인
                                ax_main.axhline(ent_price, color='yellow', linewidth=2, label=f'Entry: {ent_price:.6f}')
                                ax_main.axhline(sl_price, color='red', linestyle='--', linewidth=2, label=f'SL: {sl_price:.6f}')
                                ax_main.axhline(tp_price, color='green', linestyle='-.', linewidth=2, label=f'TP: {tp_price:.6f}')
                                
                                # 신호 발생 지점 표시
                                signal_idx = len(df_window.iloc[-30:]) - 1
                                if position == LONG:
                                    ax_main.scatter(signal_idx, ent_price, color='lime', s=150, marker="^", 
                                                label="LONG ENTRY", zorder=5)
                                else:
                                    ax_main.scatter(signal_idx, ent_price, color='red', s=150, marker="v", 
                                                label="SHORT ENTRY", zorder=5 )
                                
                                # 결과 표시
                                if hit_index is not None:
                                    result_idx = signal_idx + hit_index + 1
                                    if tp_close:
                                        ax_main.scatter(result_idx, tp_price, color='green', s=150, marker="*", 
                                                    label="TP HIT", zorder=5 )
                                    elif sl_close:
                                        ax_main.scatter(result_idx, sl_price, color='red', s=150, marker="x", 
                                                    label="SL HIT", zorder=5 )
                                
                                # 수직선으로 신호 지점 구분
                                ax_main.axvline(signal_idx, color='orange', linestyle=':', alpha=0.7, linewidth=2)
                                
                                # 타이틀 및 범례
                                title = f"{sym} {tf} - {res['pattern']}\nPnL: {pnl:+.2f}% | Conf: {res['confidence']:.0%} | Window: {window_idx}"
                                ax_main.set_title(title, fontsize=14, pad=20)
                                ax_main.legend(loc='upper left', fontsize=10)
                                ax_main.grid(True, alpha=0.3)
                                
                                # 볼륨 차트
                                vol_data = plot_data["volume"]
                                colors = ['red' if i >= signal_idx else 'blue' for i in range(len(vol_data))]
                                ax_vol.bar(plot_data["Index"], vol_data, color=colors, alpha=0.7)
                                ax_vol.set_title('Volume', fontsize=12)
                                ax_vol.grid(True, alpha=0.3)
                                
                                # 이미지 저장
                                now = datetime.now().strftime("%m%d_%H%M%S")
                                pattern_clean = res['pattern'].replace('(', '').replace(')', '').replace(' ', '_')
                                filename = f"backtest/{now}_{tf}_w{window_idx:04d}_{pattern_clean}_PnL{pnl:+.1f}%.jpg"
                                os.makedirs("backtest/", exist_ok=True)
                                plt.savefig(filename, dpi=300, bbox_inches='tight')
                                plt.close()
                            
                            # 결과 통계 업데이트
                            pattern_key = res["pattern"]
                            if tp_close:
                                if k+"_"+pattern_key + "_profit" in results.keys():
                                    results[k+"_"+pattern_key + "_profit"] += 1
                                else:
                                    results[k+"_"+pattern_key + "_profit"] = 1
                            elif sl_close:
                                if k+"_"+pattern_key + "_loss" in results.keys():
                                    results[k+"_"+pattern_key + "_loss"] += 1
                                else:
                                    results[k+"_"+pattern_key + "_loss"] = 1
                            
                            # 수익률 누적
                            slippage_rate = 0.0005  # 예: 0.05%
                            realistic_pnl = (pnl - 0.02 - slippage_rate*100) * 0.01
                            pnls *= (1 + realistic_pnl)
                            trade_count += 1
                            
                            if hit_index is not None:
                                running_times.append(hit_index)
                            
                            profit = (1 + realistic_pnl) * 100 - 100
                            # cum_profit += profit
                            
                            if k in coef_test_results.keys():
                                coef_test_results[k].append(profit)
                            else:
                                coef_test_results.update({k: [profit]})   
                                
                            # print(f"💰  Cumulative PnL: << {pnls:.2f} >>")
                            # print(f"    Cumulative profit: << {cum_profit:.2f} >>")
                            print(f"📈 Trade Count: {trade_count}")
                            if running_times:
                                print(f"⏱️ Avg Running Time: {np.mean(running_times):.1f} candles")
                            print("-" * 60)
                            
                        else:
                            print("⚠️ Not enough future data for validation")
                           
                # 조기 종료 조건
                if trade_count >= 3000:  # 최대 거래 수 제한
                    print(f"\n🛑 Early termination - Trade count: {trade_count}, Signal count: {signal_count}")
                    break
                
            # 심볼별 결과 요약
            if signal_count > 0:
                print(f"\n📊 {sym} {tf}_Summary:")
                print(f"Signals detected: {signal_count}")
                print(f"Trades executed: {trade_count}")
                print_results_simple(results)
                
            result_txt = "\n------------------------\n"
            for k in coef_test_results.keys():
                sum_profit = np.sum(coef_test_results[k])
                result_txt += f"{k}: {sum_profit}\n"
                
            print(result_txt)
    
            # 전체 백테스트 시간 제한 (3일 상당)
            total_time = np.sum(running_times) * int(tf.replace("m", ""))
            if total_time > 60 * 24 * 300:  # 3일 초과시 종료
                print(f"\n⏰ Time limit reached: {total_time/(60*24):.1f} days")
                break
            
            if trade_count >= 3000:  
                break
            if profit < -20:
                break
                
        except Exception as e:
            print(f"❌ Error processing {sym}: {str(e)}")
            try:
                await binance.close()
            except:
                pass
            continue
        
    result_txt = "\n------------------------\n"
    for k in coef_test_results.keys():
        sum_profit = np.sum(coef_test_results[k])
        result_txt += f"{k}: {sum_profit}\n"
        
    print(result_txt)
    
    with open("backtest_result.txt", 'a') as f:
        f.write(str(result_txt))      
        
    return profit
    
    
def print_results_simple(results):
    if not results:
        return
    
    patterns = {}
    for key, value in results.items():
        pattern = key.replace('_profit', '').replace('_loss', '').replace('(', '').replace(')', '')
        if pattern not in patterns:
            patterns[pattern] = [0, 0]  # [wins, losses]
            
        if '_profit' in key:
            patterns[pattern][0] = value
        else:
            patterns[pattern][1] = value
            
    print("📊 Results:")
    for pattern, (w, l) in patterns.items():
        wr = w/(w+l)*100 if w+l > 0 else 0
        emoji = "🟢" if wr >= 50 else "🔴"
        print(f"  {emoji} {pattern}: {w}W/{l}L ({wr:.0f}%)")



if __name__ == "__main__":

    tf = "15m"
    profit = asyncio.run(run_sliding_backtest(tf, T_window=70, T_start=71, T_end=1400))
          