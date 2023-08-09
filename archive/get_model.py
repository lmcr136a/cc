# import torch
import numpy as np
# from torch import datasets, transforms
from bull_bear import past_data
from HYPERPARAMETERS import *
from deep.dataloader import minmax_for_dataloader

pretrained_path = './deep/best_model.pt'

def get_model_prediction(binance, sym):
    n = 48
    # default_trans = transforms.ToTensor()
    df3 = past_data(binance, sym=sym, tf='3m', limit=1500)['close'][-n:]
    df15 = past_data(binance, sym=sym, tf='15m', limit=1500)['close'][-n:]
    input_ = np.concatenate([np.expand_dims(df3, 0), np.expand_dims(df15, 0)], axis=0)
    input_ = minmax_for_dataloader(input_)
    input_ = np.expand_dims(input_, 0)  # 1, 2, 48
    input_ = np.expand_dims(input_, 0)  # 1, 1, 2, 48
    # input_ = default_trans(input_)

    # net = torch.load(pretrained_path, map_location=torch.device('cpu'))

    # X = torch.tensor(input_, dtype=net.conv1.weight.dtype)
    # pred = net(X)[0]

    # p = list(map(lambda x: round(x, 1), pred.data.tolist()))
    # print(f"[{sym}: {p}]", end='')
    # pred = torch.argmax(pred)

    # if pred == 0:           # 아무것도아님
    #     return None
    # elif pred == 1:
    #     return LONG
    # elif pred == 2:
    #     return SHORT


# if __name__ == "__main__":
#     from utils import *
#     binance = get_binance()
#     sym = "GRT/USDT"
#     get_model_prediction(binance, sym)