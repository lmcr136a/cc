import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf 
from mplfinance.original_flavor import candlestick_ohlc


def check_trend_line(support: bool, pivot: int, slope: float, y: np.array):
    # compute sum of differences between line and prices, 
    # return negative val if invalid 
    
    # Find the intercept of the line going through pivot point with given slope
    intercept = -slope * pivot + y[pivot]
    line_vals = slope * np.arange(len(y)) + intercept
     
    diffs = line_vals - y
    
    # Check to see if the line is valid, return -1 if it is not valid.
    if support and diffs.max() > 1e-5:
        return -1.0
    elif not support and diffs.min() < -1e-5:
        return -1.0

    # Squared sum of diffs between data and line 
    err = (diffs ** 2.0).sum()
    return err;


def optimize_slope(support: bool, pivot:int , init_slope: float, y: np.array):
    
    # Amount to change slope by. Multiplyed by opt_step
    slope_unit = (y.max() - y.min()) / len(y) 
    
    # Optmization variables
    opt_step = 1.0
    min_step = 0.0001
    curr_step = opt_step # current step
    
    # Initiate at the slope of the line of best fit
    best_slope = init_slope
    best_err = check_trend_line(support, pivot, init_slope, y)
    if not best_err >= 0.0: # Shouldn't ever fail with initial slope
        return (np.nan, np.nan)
    get_derivative = True
    derivative = None
    while curr_step > min_step:

        if get_derivative:
            # Numerical differentiation, increase slope by very small amount
            # to see if error increases/decreases. 
            # Gives us the direction to change slope.
            slope_change = best_slope + slope_unit * min_step
            test_err = check_trend_line(support, pivot, slope_change, y)
            derivative = test_err - best_err;
            
            # If increasing by a small amount fails, 
            # try decreasing by a small amount
            if test_err < 0.0:
                slope_change = best_slope - slope_unit * min_step
                test_err = check_trend_line(support, pivot, slope_change, y)
                derivative = best_err - test_err

            if test_err < 0.0: # Derivative failed, give up
                raise Exception("Derivative failed. Check your data. ")

            get_derivative = False

        if derivative > 0.0: # Increasing slope increased error
            test_slope = best_slope - slope_unit * curr_step
        else: # Increasing slope decreased error
            test_slope = best_slope + slope_unit * curr_step
        

        test_err = check_trend_line(support, pivot, test_slope, y)
        if test_err < 0 or test_err >= best_err: 
            # slope failed/didn't reduce error
            curr_step *= 0.5 # Reduce step size
        else: # test slope reduced error
            best_err = test_err 
            best_slope = test_slope
            get_derivative = True # Recompute derivative
    
    # Optimize done, return best slope and intercept
    return (best_slope, -best_slope * pivot + y[pivot])


def fit_trendlines_single(data: np.array):
    # find line of best fit (least squared) 
    # coefs[0] = slope,  coefs[1] = intercept 
    x = np.arange(len(data))
    coefs = np.polyfit(x, data, 1)

    # Get points of line.
    line_points = coefs[0] * x + coefs[1]

    # Find upper and lower pivot points
    upper_pivot = (data - line_points).argmax() 
    lower_pivot = (data - line_points).argmin() 
   
    # Optimize the slope for both trend lines
    support_coefs = optimize_slope(True, lower_pivot, coefs[0], data)
    resist_coefs = optimize_slope(False, upper_pivot, coefs[0], data)

    return (support_coefs, resist_coefs) 



def fit_trendlines_high_low(high: np.array, low: np.array, close: np.array):
    x = np.arange(len(close))
    coefs = np.polyfit(x, close, 1)
    line_points = coefs[0] * x + coefs[1]
    
    upper_pivot = (high - line_points).argmax() 
    lower_pivot = (low - line_points).argmin() 
    
    support_coefs = optimize_slope(True, lower_pivot, coefs[0], low)
    resist_coefs = optimize_slope(False, upper_pivot, coefs[0], high)

    return (support_coefs, resist_coefs)


def save_trendline_img(candles, imgfilename):
    support_coefs, resist_coefs = fit_trendlines_high_low(candles['high'], candles['low'], candles['close'])
    support_line = support_coefs[0] * np.arange(len(candles)) + support_coefs[1]
    resist_line = resist_coefs[0] * np.arange(len(candles)) + resist_coefs[1]

    plt.style.use('dark_background')
    ax = plt.gca()
    ax.grid(False)
    ax.set_facecolor((0.9, 0.9, 0.85))

    x = candles.index
    ax.plot(support_line, color="b")
    ax.plot(resist_line, color="b")
    
    candles.loc[:,"Index"] = candles.index
    candlestick_ohlc(ax, candles.loc[:, ["Index", "open", "high", "low", "close"] ].values, width=0.6, colorup='green', colordown='red', alpha=0.8)
    ax.scatter(candles["Index"], candles["PointPos"], color="b")

    os.makedirs("Figures/", exist_ok=True)
    plt.savefig(f"Figures/{imgfilename}.jpg")
    plt.close()
    

def check_line_fit(slope: float, y: np.array, err_ratio=0.003):
    price = y[0]
    line_vals = slope * np.arange(len(y)) + price
     
    diffs = line_vals - y
    if diffs.max() > price*err_ratio or diffs.min() < -price*err_ratio:
        return False
    return True



def find_t_convergence(pivots, pivottype, interval, ref_price):
    support_idx = np.where(pivottype == 1)[0]  # low
    s_ps_df = pivots.loc[support_idx]
    resist_idx = np.where(pivottype == 2)[0]   # high
    r_ps_df = pivots.loc[resist_idx]
    
    s_ps = np.array(s_ps_df)
    r_ps = np.array(r_ps_df)
    
    x = list(range(interval))
    s, r = None, None
    # bottom
    i = len(s_ps)-interval
    block_data = s_ps[i: i+interval]
    c = np.polyfit(x, block_data, 1)
    if check_line_fit(c[0], block_data):
        s = {"a": c[0] / (support_idx[i+interval-1] - support_idx[i])*(interval-1), 
                        "start_x": support_idx[i], 
                        "last_x1": support_idx[i+interval-3], 
                        "last_x2": support_idx[i+interval-2], 
                        "last_x3": support_idx[i+interval-1], 
                        "b": c[1]}
    
    # high
    i = len(r_ps) - interval
    block_data = r_ps[i: i+interval]
    c = np.polyfit(x, block_data, 1)
    if check_line_fit(c[0], block_data):
        r = {"a": c[0] / (resist_idx[i+interval-1] - resist_idx[i])*(interval-1), 
                        "start_x": resist_idx[i], 
                        "last_x1": resist_idx[i+interval-3], 
                        "last_x2": resist_idx[i+interval-2], 
                        "last_x3": resist_idx[i+interval-1], 
                        "b": c[1]}
    if not s or not r:
        return (None, None)
    
    max_slope_diff = ref_price*0.01*0.3
    min_slope_diff = ref_price*0.01*0.01
    x_index_ref = 20
    
    if np.abs(s["start_x"] - r['start_x']) < x_index_ref and \
        np.abs(s["last_x3"] - r['last_x3']) < x_index_ref and \
        (s["a"] > r["a"]) and \
        np.abs(s["a"] - r["a"]) > min_slope_diff and \
        np.abs(s["a"] - r["a"]) < max_slope_diff:
       return (s, r)
   
    return (None, None)


def check_trend(df):
    ppos_ind_list = df[df['PointPos'] > 0].index
    pivots = df.loc[ppos_ind_list]
    pivot_vals_high = np.array(pivots['high'])
    pivot_vals_low = np.array(pivots['low'])
    
    small_grads_high = pivot_vals_high[1:]-pivot_vals_high[:-1]  # diff btw each pivots
    small_grads_low = pivot_vals_low[1:]-pivot_vals_low[:-1]  # diff btw each pivots
    block=6
    block_grads = []
    blocks_ind = []
    for i in range(len(small_grads_high)-block+1):
       block_grads.append(np.mean(small_grads_high[i:i+block]+small_grads_low[i:i+block]))
       blocks_ind.append([ppos_ind_list[i], ppos_ind_list[i+block]])
       
    assert len(pivot_vals_high) - block == len(block_grads) == len(blocks_ind)
    assert blocks_ind[-1][1] == ppos_ind_list[-1]
    
    ref = 0.5*(max(df["close"])+ min(df["close"]))*0.01*0.5
    def bull_or_bear(block_grad, ref=ref):
        if block_grad > ref:
            return 2
        elif block_grad < -ref:
            return 1
        else:
            return 0
    
    def no_cons_trend(li):
        for i in range(len(li)-1):
            if li[i] == li[i+1]:
                return False
        return True
    blocks = list(map(bull_or_bear, block_grads))
    
    
    while not no_cons_trend(blocks):
        for b_i in range(len(blocks)-1):
            if b_i < len(blocks)-1:
                if blocks[b_i] == blocks[b_i+1]:
                    blocks.pop(b_i+1)
                    blocks_ind[b_i][1] = blocks_ind[b_i+1][1]
                    blocks_ind.pop(b_i+1)
        for b_i in range(len(blocks)-2):
            if b_i < len(blocks)-2:
                if blocks[b_i] == blocks[b_i+2] and blocks[b_i+1] == 0:
                    blocks.pop(b_i+2)
                    blocks.pop(b_i+1)
                    blocks_ind[b_i][1] = blocks_ind[b_i+2][1]
                    blocks_ind.pop(b_i+2)
                    blocks_ind.pop(b_i+1)
                    
    while 0 in blocks[1:]:
        for b_i in range(1, len(blocks)):
            if b_i < len(blocks):
                if blocks[b_i] == 0:
                    blocks.pop(b_i)
                    blocks_ind[b_i-1][1] = blocks_ind[b_i][1]
                    blocks_ind.pop(b_i)
                    
    trends_by_x = np.zeros(len(df))
    for b_i in range(len(blocks)):
        trends_by_x[blocks_ind[b_i][0]:blocks_ind[b_i][1]] += blocks[b_i]
    trends_by_x = np.where(trends_by_x == 3, 0, trends_by_x)
    # print(blocks)
    # print(blocks_ind)
    return blocks, blocks_ind, trends_by_x
    
    
def begining_of_trend(trends_by_x, pointpos):
    return False
    