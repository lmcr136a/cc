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
    

async def find_support_resistance_window(df_window, sym, tf="3m", imgfilename="realtime", window_idx=0):
    df = df_window.copy()
    df.reset_index(drop=True, inplace=True)

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
    
    # 볼린저 밴드
    period = min(15, len(df)-1)
    df["bb_middle"] = close.rolling(period).mean()
    df["bb_std"] = close.rolling(period).std()
    df["bb_upper"] = df["bb_middle"] + (df["bb_std"] * 1.5)  # 표준편차 축소
    df["bb_lower"] = df["bb_middle"] - (df["bb_std"] * 1.5)
    
    # MACD (간단화)
    if len(df) > 12:
        ema_fast = close.ewm(span=8).mean()
        ema_slow = close.ewm(span=16).mean()
        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=5).mean()
    else:
        df["macd"] = 0
        df["macd_signal"] = 0
    
    # VWAP
    df["vwap"] = (close * volume).cumsum() / volume.cumsum()
    
    # 현재 상태 (마지막 값)
    curr_price = close.iloc[-1]
    curr_rsi = df["rsi"].iloc[-1]
    curr_ema5 = df["ema_5"].iloc[-1]
    curr_ema10 = df["ema_10"].iloc[-1]
    curr_ema20 = df["ema_20"].iloc[-1]
    curr_vol = volume.iloc[-1]
    curr_vol_avg = df["vol_avg"].iloc[-1]
    curr_bb_upper = df["bb_upper"].iloc[-1]
    curr_bb_lower = df["bb_lower"].iloc[-1]
    curr_bb_middle = df["bb_middle"].iloc[-1]
    curr_macd = df["macd"].iloc[-1]
    curr_macd_signal = df["macd_signal"].iloc[-1]
    curr_vwap = df["vwap"].iloc[-1]
    
    # 최근 고점/저점 (간단하게)
    recent_high = high.iloc[-min(20, len(df)):].max()
    recent_low = low.iloc[-min(20, len(df)):].min()
    
    # 기본 변수 초기화
    pattern = None
    ent_price1, tp_price1, sl_price1, curr_price1 = None, None, None, False
    ent_price2, tp_price2, sl_price2, curr_price2 = None, None, None, False
    confidence = 0
    
    # 스캘핑용 리스크 관리 (완화)
    reward_ratio = 1.5
    
    # 5. 볼린저밴드 터치 (매우 단순)
    if not curr_price1 and not curr_price2:
        bb_touch_lower = curr_price <= curr_bb_lower * 1.005  # 하단 근접
        bb_touch_upper = curr_price >= curr_bb_upper * 0.995  # 상단 근접
        
        if bb_touch_lower and curr_rsi < 60 and curr_bb_lower * 0.995 < curr_price:
            ent_price1 = curr_price
            sl_price1 = min(curr_bb_lower * 0.995, curr_price * 0.997)
            tp_price1 = max(curr_bb_middle, curr_price + (curr_price - sl_price1) * reward_ratio)
            curr_price1 = True
            pattern = "Bollinger Lower Touch"
            confidence = 0.55
        
        elif bb_touch_upper and curr_rsi > 40 and curr_price < curr_bb_upper * 1.005:
            ent_price2 = curr_price
            sl_price2 = max(curr_bb_upper * 1.005, curr_price * 1.003)
            tp_price2 = min(curr_bb_middle, curr_price - (sl_price2 - curr_price) * reward_ratio)
            curr_price2 = True
            pattern = "Bollinger Upper Touch"
            confidence = 0.55
    
    # 8. 폴백 전략 - 거의 무조건 신호 (데이터 있으면)
    if not curr_price1 and not curr_price2 and len(df) >= 2:
        # 마지막 2봉 비교해서 방향 결정
        if close.iloc[-1] > close.iloc[-2]:  # 상승
            ent_price1 = curr_price
            sl_price1 = curr_price * 0.996  # 0.5% 손절
            tp_price1 = curr_price * 1.006  # 0.6% 익절
            curr_price1 = True
            pattern = "Fallback Long"
            confidence = 0.40
        else:  # 하락
            ent_price2 = curr_price
            sl_price2 = curr_price * 1.004  # 0.5% 손절
            tp_price2 = curr_price * 0.994  # 0.6% 익절
            curr_price2 = True
            pattern = "Fallback Short"
            confidence = 0.40
    
    if not pattern:
        return None
    
    # 차트 플롯 및 저장
    if pattern and 0:
        RED, ORANGE, YELLOW, GREEN = (0.6, 0, 0, 1), (0.7, 0.5, 0, 1), (0.6, 0.6, 0, 1), (0.1, 0.5, 0, 1)
        BLUE, PURPLE = (0.2, 0.3, 0.8), (0.5, 0.1, 0.6)
        plt.rcParams["figure.figsize"] = (12, 12)
        f, (ax, ax_vol, ax_rsi, ax_stoch, ax_channel) = plt.subplots(5, 1, gridspec_kw={'height_ratios': [3, 1, 1, 1, 1]})
        ax.set_facecolor((0.95, 0.95, 0.9))
        plt.subplots_adjust(top=0.9, bottom=0.05, right=0.98, left=0.1, hspace=0.5)

        candlestick_ohlc(ax, df.loc[:, ["Index", "open", "high", "low", "close"]].values, width=0.6, colorup='green', colordown='red', alpha=0.8)
        
        title = f"{pattern} - {sym} {tf} (Window {window_idx})\nConf: {confidence:.0%} | Trend: {trend_regime} | Vol: {volatility_regime} | R:R = 1:{rr_ratio:.1f}"
        ax.set_title(title, position=(0.5, 1.05), fontsize=14)
        
        # 핵심 지표만 표시
        ax.plot(df["Index"], df["ema_fast"], color=RED, label=f'EMA Fast({9})', linewidth=2)
        ax.plot(df["Index"], df["ema_slow"], color=ORANGE, label=f'EMA Slow({21})', linewidth=1.5)
        ax.plot(df["Index"], df["vwap"], color=PURPLE, linestyle='--', label='VWAP', linewidth=2)
        ax.plot(df["Index"], df["bb_upper"], color=YELLOW, linestyle='--', label='BB', alpha=0.8)
        ax.plot(df["Index"], df["bb_lower"], color=YELLOW, linestyle='--', alpha=0.8)
        ax.plot(df["Index"], df["high_channel"], color='gray', linestyle=':', alpha=0.6, label='Channel')
        ax.plot(df["Index"], df["low_channel"], color='gray', linestyle=':', alpha=0.6)
        ax.legend(loc='upper left', fontsize=10)
        
        # 볼륨 (볼륨 비율 색상 코딩)
        colors = ['darkred' if ratio > 1.5 else 'red' if ratio > 1.2 else 'orange' if ratio > 1.0 else 'gray' 
                 for ratio in df["volume_ratio"]]
        ax_vol.bar(df["Index"], df["volume"], color=colors, alpha=0.7)
        ax_vol.axhline(df["volume_ma"].iloc[-1], color='blue', linestyle='--', alpha=0.7)

        # RSI 차트
        ax_rsi.plot(df["Index"], df["rsi"], color=BLUE, label='RSI(14)', linewidth=2)
        ax_rsi.plot(df["Index"], df["rsi_fast"], color='orange', label='RSI(7)', linewidth=1.5)
        ax_rsi.axhline(30, ls='--', c='g', alpha=0.7)
        ax_rsi.axhline(70, ls='--', c='r', alpha=0.7)
        ax_rsi.axhline(25, ls=':', c='g', alpha=0.5)
        ax_rsi.axhline(75, ls=':', c='r', alpha=0.5)
        ax_rsi.set_title(f'RSI (Current: {curr_rsi:.1f})', fontsize=12)
        ax_rsi.legend(fontsize=10)
        
        # 스토캐스틱
        ax_stoch.plot(df["Index"], df["stoch_k"], color='blue', label='%K', linewidth=2)
        ax_stoch.plot(df["Index"], df["stoch_d"], color='red', label='%D', linewidth=1.5)
        ax_stoch.axhline(20, ls='--', c='g', alpha=0.7)
        ax_stoch.axhline(80, ls='--', c='r', alpha=0.7)
        ax_stoch.set_title(f'Stochastic (K: {curr_stoch_k:.1f}, D: {curr_stoch_d:.1f})', fontsize=12)
        ax_stoch.legend(fontsize=10)
        
        # 채널 포지션
        ax_channel.plot(df["Index"], df["channel_position"], color='purple', linewidth=2)
        ax_channel.axhline(0.2, ls='--', c='g', alpha=0.7, label='Low')
        ax_channel.axhline(0.8, ls='--', c='r', alpha=0.7, label='High')
        ax_channel.axhline(0.5, ls='-', c='gray', alpha=0.5, label='Mid')
        ax_channel.set_title(f'Channel Position ({curr_channel_pos:.2f})', fontsize=12)
        ax_channel.legend(fontsize=10)
        
        # 진입/손절/익절 포인트
        if ent_price1:
            ax.scatter(curr_idx, ent_price1, color='limegreen', s=100, marker="^", label="LONG", zorder=5)
            ax.axhline(sl_price1, color='red', linestyle=':', alpha=0.8, linewidth=2)
            ax.axhline(tp_price1, color='green', linestyle=':', alpha=0.8, linewidth=2)
            
        if ent_price2:
            ax.scatter(curr_idx, ent_price2, color='red', s=100, marker="v", label="SHORT", zorder=5)
            ax.axhline(sl_price2, color='red', linestyle=':', alpha=0.8, linewidth=2)
            ax.axhline(tp_price2, color='green', linestyle=':', alpha=0.8, linewidth=2)

        # 리스크 정보 텍스트
        risk_text = f"Risk: {dynamic_stop_pct*100:.2f}% | ATR: {curr_atr:.6f}"
        ax.text(0.02, 0.98, risk_text, transform=ax.transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        os.makedirs("sliding_backtest/", exist_ok=True)
        now = datetime.now().strftime("%m%d_%H%M%S")
        plt.savefig(f"sliding_backtest/{now}_{imgfilename}_w{window_idx:04d}_{pattern.replace('(', '').replace(')', '')}.jpg", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
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


async def run_sliding_backtest(tf, T_window, T_start=100, T_end=500, T_step=10):
    
    profit = 0
    pnls = 100
    running_times = []
    trade_count = 0
    results = {}
    signal_count = 0
    
    large_vol_symlist = []
    random.shuffle(SYMLIST)
    
    for i, sym in enumerate(SYMLIST):
        try:
            print(f"\n=== Testing {sym} ===")
            
            # 볼륨 체크
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
            # 전체 데이터 한번만 호출 (충분히 큰 limit)
            future_t = 100
            total_limit = T_end + future_t  # 여유분 추가
            df_full = await past_data(sym, tf, total_limit)
            df_full.reset_index(drop=True, inplace=True)
            
            print(f"Full data loaded: {len(df_full)} candles")
            
            # 슬라이딩 윈도우로 백테스트
            for window_idx, t in enumerate(range(T_start, min(T_end, len(df_full)-future_t), T_step)):
                # 윈도우 데이터 추출 (t개 캔들)
                df_window = df_full.iloc[t-T_window:t].copy()
                
                if len(df_window) < 60:  # 최소 데이터 요구사항
                    continue
                
                # 시그널 감지
                res = await find_support_resistance_window(
                    df_window, sym, tf, 
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
                        
                        highs, lows = np.array(future_df["high"]), np.array(future_df["low"])
                        tp_close, sl_close, pnl = False, False, 0
                        hit_index = None
                        
                        # 포지션별 손익 계산
                        if position == LONG:
                            tp_is = np.where(highs >= tp_price)[0]
                            sl_is = np.where(lows <= sl_price)[0]
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
                                pnl = 0
                                print(f"⏸️ NO CONCLUSION - Current PnL: {pnl:.2f}%")
                                
                        elif position == SHORT:
                            tp_is = np.where(lows <= tp_price)[0]
                            sl_is = np.where(highs >= sl_price)[0]
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
                                pnl = 0
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
                            if pattern_key + "_profit" in results:
                                results[pattern_key + "_profit"] += 1
                            else:
                                results[pattern_key + "_profit"] = 1
                        elif sl_close:
                            if pattern_key + "_loss" in results:
                                results[pattern_key + "_loss"] += 1
                            else:
                                results[pattern_key + "_loss"] = 1
                        
                        # 수익률 누적
                        pnls *= (1 + (pnl - 0.02) * 0.01)  # 0.02% 수수료 차감
                        trade_count += 1
                        
                        if hit_index is not None:
                            running_times.append(hit_index)
                        
                        profit += (1 + (pnl - 0.02) * 0.01) * 100 - 100
                        
                        print(f"💰 Cumulative PnL: << {pnls:.2f} >>")
                        print(f"💰 Cumulative profit: << {profit:.2f} >>")
                        print(f"📈 Trade Count: {trade_count}")
                        if running_times:
                            print(f"⏱️ Avg Running Time: {np.mean(running_times):.1f} candles")
                        print("-" * 60)
                        
                    else:
                        print("⚠️ Not enough future data for validation")
                        
                # 조기 종료 조건
                if trade_count >= 10000 or signal_count >= 10000:  # 최대 거래 수 제한
                    print(f"\n🛑 Early termination - Trade count: {trade_count}, Signal count: {signal_count}")
                    break
                
            # 심볼별 결과 요약
            if signal_count > 0:
                print(f"\n📊 {sym} {tf} Summary:")
                print(f"Signals detected: {signal_count}")
                print(f"Trades executed: {trade_count}")
                print_results_simple(results)
                
            # 전체 백테스트 시간 제한 (3일 상당)
            total_time = np.sum(running_times) * int(tf.replace("m", ""))
            if total_time > 60 * 24 * 300:  # 3일 초과시 종료
                print(f"\n⏰ Time limit reached: {total_time/(60*24):.1f} days")
                break
                
        except Exception as e:
            print(f"❌ Error processing {sym}: {str(e)}")
            try:
                await binance.close()
            except:
                pass
            continue
    # with open("large_vol_symlist.txt", "w") as f:
    #     f.write(str(large_vol_symlist))
    
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

    tf = "3m"
    pnls = []
    asyncio.run(run_sliding_backtest(tf, T_window=300, T_start=301, T_end=1400))