
import numpy as np
import pandas as pd
import mplfinance as mpf
import time
import random
import matplotlib.pyplot as plt
from scipy import io
from utils import *
"""
마지막 수정: 2023 07 24
"""

test_num = 20000

filename_total =  "total_dataset.mat"
filename_train =  "train_dataset.mat"
filename_test =  "test_dataset.mat"
# dataset = io.loadmat(filename_total)

binance = get_binance()
n = 48
label_n = 10
ref = 0.55

data = pd.DataFrame(columns = ['inputs', 'label'])

dataset = {'inputs': [], 'labels': []} #################################################3
labelinfo = {"0":0, "1":0, "2":0}       # 0: 아무것도아님 1:상승 2:하락
for sym in SYMLIST:
    df3 = past_data(binance, sym=sym, tf='3m', limit=1500)['close']
    df15 = past_data(binance, sym=sym, tf='15m', limit=1500)['close']
    for idx in range(1, len(df3)-n-label_n):
        input_3 = df3[-(idx+n+label_n):-(idx+label_n)]
        idx15 = (idx+label_n)//5+1
        input_15 = df15[-(idx15+n):-idx15-1]
        input_15 = pd.concat([input_15, input_3[-1:]])

        label_vec = df3[-(idx+label_n):-idx]
        future_low, future_high, past = min(label_vec), max(label_vec), label_vec[0]
        diff_ratio_l = (future_low - past)/past*100
        diff_ratio_h = (future_high - past)/past*100

        if abs(diff_ratio_l) > abs(diff_ratio_h) and diff_ratio_l < -ref:
            label = 2
            labelinfo['2'] += 1
        elif abs(diff_ratio_l) < abs(diff_ratio_h) and diff_ratio_h > ref:
            label = 1
            labelinfo['1'] += 1
        else:
            label = 0
            if labelinfo['0'] > labelinfo['1'] or labelinfo['0'] > labelinfo['2']:
                continue
            labelinfo['0'] += 1

        input_ = np.concatenate([np.expand_dims(input_3, 0), np.expand_dims(input_15, 0)], axis=0)
        dataset['inputs'].append(input_.tolist())
        dataset['labels'].append(label)
    print(labelinfo)
dataset['inputs'] = np.array(dataset['inputs'])
dataset['labels'] = np.array(dataset['labels'])
if len(dataset['labels']) < 3:
    dataset['labels'] = dataset['labels'][0]
num = len(dataset['labels'])
print(dataset['labels'])
mat_file = io.savemat(filename_total, dataset)

print("total input shape: ", dataset['inputs'].shape)
print("total label shape: ", dataset['labels'].shape)
mat_file = io.savemat(filename_total, dataset)


shuffleidx = [n for n in range(num)]
random.shuffle(shuffleidx)
testidx = shuffleidx[:test_num]
trainidx =shuffleidx[test_num:]

train_dataset = {'inputs': dataset['inputs'][trainidx], 'labels': dataset['labels'][trainidx]}
test_dataset = {'inputs': dataset['inputs'][testidx], 'labels': dataset['labels'][testidx]}

mat_file = io.savemat(filename_train, train_dataset)
mat_file = io.savemat(filename_test, test_dataset)

print("이거 이제 옮겨라")