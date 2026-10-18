import numpy as np
import csv
from collections import defaultdict
from ss1 import ssvepDetect

srate = 250
dataLen = 4
freqs = [8, 9, 10, 11, 12, 13, 14, 15]
freq_labels = [8, 9, 10, 11, 12, 13, 14, 15]

for filename in ['D1.csv', 'D2.csv']:
    datapath = r'C:/Users/Administrator/Desktop/v5/ExampleData/' + filename
    sd = ssvepDetect(srate, freqs, dataLen)

    data = []
    with open(datapath, mode='r') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)
        for row in csv_reader:
            data.append([float(_) for _ in row])
    data = np.array(data, dtype=np.float64)
    points = dataLen * srate

    # 构建全部试次数据（按 freq_idx 组织）
    all_epochs = []
    all_stims = []
    train_data = defaultdict(list)
    for i in range(48):
        epoch = data[i*points:(i+1)*points, :6].T
        stim = int(data[i*points, -1])
        all_epochs.append(epoch)
        all_stims.append(stim)
        train_data[stim].append(epoch)

    # LOO 评估：对每个试次，排除自身后训练，再测试
    results = []
    for i in range(48):
        stim = all_stims[i]
        epoch = all_epochs[i]

        # LOO 训练数据
        loo_train = defaultdict(list)
        trial_counters = defaultdict(int)
        for j in range(48):
            if j == i:
                continue
            s = all_stims[j]
            loo_train[s].append(all_epochs[j])

        res = sd.detect(epoch, train_data=loo_train)
        results.append(res)

    acc = sum(1 for r, s in zip(results, all_stims) if r == s) / 48
    print("%s accuracy: %.4f (%d/48)" % (filename, acc, int(acc * 48)))
    for i, (r, s) in enumerate(zip(results, all_stims)):
        if r != s:
            print("  trial %d: pred=%d(%dHz) true=%d(%dHz)" % (
                i, r, freq_labels[r], s, freq_labels[s]))
