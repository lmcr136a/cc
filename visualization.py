import datetime
import numpy as np
import pandas as pd
import mplfinance as mpf
import time
import matplotlib.pyplot as plt
import ccxt 


def show_default_graph(df, m1=None, m2=None, m3=None, n=100, title=''):
    plt.style.use('fivethirtyeight')
    df = df[-n:]
    fig, ax = plt.subplots(figsize = (12,6))

    mpf.plot(df,type='candle', volume=False, style='charles', ax=ax)
    xpoints = np.arange(len(df))

    if str(type(m1)) != "<class 'NoneType'>":
        ax.plot(xpoints, m1[-n:], color = (1, 0.3, 0.2), linewidth = 1, label='Close, 5-Day SMA')
    if str(type(m2)) != "<class 'NoneType'>":
        ax.plot(xpoints, m2[-n:], color = (0.5, 0.2, 0.7), linewidth = 1, label='Close, 20-Day SMA')
    if str(type(m3)) != "<class 'NoneType'>":
        ax.plot(xpoints, m3[-n:], color = (0.2, 0, 1), linewidth = 1, label='Close, 40-Day SMA')

    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.legend()

    ax.set_title(title)
    plt.savefig("graph.png")
    plt.close()



def get_default_graph(m1, m2=None, m3=None, m4=None, n=100, title=''):
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots(figsize = (12,6))

    n = len(m1)
#     mpf.plot(df,type='candle', volume=False, style='charles', ax=ax)
    xpoints = np.arange(len(m1))[-n:]

    if str(type(m1)) != "<class 'NoneType'>":
        ax.plot(xpoints, m1[-n:], color = (1, 0.7, 0), linewidth = 2, label='Close, 5-Day SMA')
    if str(type(m2)) != "<class 'NoneType'>":
        ax.plot(xpoints, m2[-n:], color = (1, 0.2, 0.2), linewidth = 1, label='Close, 20-Day SMA')
    if str(type(m3)) != "<class 'NoneType'>":
        ax.plot(xpoints, m3[-n:], color = (0.2, 0.2, 1), linewidth = 1, label='Close, 40-Day SMA')
    if str(type(m4)) != "<class 'NoneType'>":
        ax.plot(xpoints, m4[-n:], color = (0.7, 0.7, 1), linewidth = 1, label='Close, 40-Day SMA')

    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')

    ax.set_title(title)
    return ax

def show_transaction(transactions, m1=None, m2=None, m3=None, m4=None, n=100, title=''):
    n = len(m1)
    if n < 200:
        R = 2
    elif n < 500:
        R = 5
    else:
        R = 10
    ax = get_default_graph(m1, m2, m3, m4)
    for tr in transactions:
        color = 'b' if tr['position'] == 'Short' else 'r'
        ax.add_patch(plt.Arrow(
            tr['ent'][0], tr['ent'][1], 
            tr['close'][0]-tr['ent'][0], tr['close'][1]-tr['ent'][1],
            width = R, color=color, zorder=2
        ))
    plt.savefig("graph.png")
    plt.close()
    return ax


