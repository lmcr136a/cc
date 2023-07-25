import torch
import numpy as np
from bull_bear import past_data
from HYPERPARAMETERS import *


pretrained_path = './deep/best_model.pt'

def get_model_prediction(binance, sym):
    n = 48
    df3 = past_data(binance, sym=sym, tf='3m', limit=n)['close']
    df15 = past_data(binance, sym=sym, tf='15m', limit=n)['close']
    input_ = np.concatenate([np.expand_dims(df3, 0), np.expand_dims(df15, 0)], axis=0)
    input_ = np.expand_dims(input_, 0)  # 1, 2, 48
    input_ = np.expand_dims(input_, 0)  # 1, 1, 2, 48

    net = torch.load(pretrained_path).to('cpu')

    X = torch.tensor(input_, dtype=net.conv1.weight.dtype)
    pred = net(X)
    pred = torch.argmax(pred[0])

    if pred == 0:           # 아무것도아님
        return 0
    elif pred == 1:
        return LONG
    elif pred == 2:
        return SHORT


# if __name__ == "__main__":
#     from utils import *
#     binance = get_binance()
#     sym = "GRT/USDT"
#     get_model_prediction(binance, sym)