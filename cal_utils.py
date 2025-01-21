
def cal_rsi(close, n=14):
    # close: pd.Series
    rsi = close.diff(1).fillna(0)
    up, down = rsi.copy(), rsi.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    gain = up.rolling(window=n).mean()
    loss = abs(down.rolling(window=n).mean())
    rs = gain/loss
    rsi = 100 - (100/(1+rs))
    return rsi
    

def cal_srsi(close, n=14, k=3, d=3):
    # close: pd.Series
    rsi = cal_rsi(close, n=n)
    sto_rsi_k = (rsi - rsi.rolling(window=k).min() / (rsi.rolling(window=k).max() - rsi.rolling(window=k).min()))
    sto_rsi_d = sto_rsi_k.rolling(window=d).mean()
    return sto_rsi_k, sto_rsi_d
    
    
    