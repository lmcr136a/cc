import pandas as pd

def cal_rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/n, min_periods=n, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi.fillna(0)

def cal_srsi(close: pd.Series, n: int = 14, k: int = 14, d: int = 3) -> tuple[pd.Series, pd.Series]:
    rsi = cal_rsi(close, n=n)

    min_rsi = rsi.rolling(window=k, min_periods=k).min()
    max_rsi = rsi.rolling(window=k, min_periods=k).max()
    
    sto_rsi_k = (rsi - min_rsi) / (max_rsi - min_rsi) * 100
    sto_rsi_k = sto_rsi_k.fillna(0)

    sto_rsi_d = sto_rsi_k.rolling(window=d, min_periods=d).mean()
    sto_rsi_d = sto_rsi_d.fillna(0)

    return sto_rsi_k, sto_rsi_d
