import numpy as np
import csv
from collections import defaultdict
from ss1 import ssvepDetect

if __name__ == '__main__':
    # 实验数据路径
    datapath = r'C:/Users/Administrator/Desktop/v5/ExampleData/D2.csv'

    # 实验参数
    srate = 250 # 采样率250Hz
    dataLen = 4 # example数据长度为4秒

    # 实例化ssvep检测器
    sd = ssvepDetect(srate,[8,9,10,11,12,13,14,15],dataLen)

    # 读取数据
    data = []
    with open(datapath, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            rowvalue = [float(_) for _ in row]
            data.append(rowvalue)
    data = np.array(data,dtype=np.float64)
    points = dataLen * srate

    # 构建全部试次数据
    all_epochs = []
    all_stims = []
    train_data = defaultdict(list)
    for i in range(48):
        epoch = data[i*points:(i+1)*points,:6].T
        stim = int(data[i*points,-1])
        all_epochs.append(epoch)
        all_stims.append(stim)
        train_data[stim].append(epoch)

    # LOO 检测
    results = []
    corr = []
    for i in range(48):
        epoch = all_epochs[i]
        # LOO：排除当前试次
        loo_train = defaultdict(list)
        for j in range(48):
            if j == i:
                continue
            loo_train[all_stims[j]].append(all_epochs[j])
        res = sd.detect(epoch, train_data=loo_train)
        results.append(res)
        corr.append(1 if res == all_stims[i] else 0)

    print("正确率： %.2f"%(sum(corr)/48))
    for i in range(48):
        print("task%d预测值：%d"%(i,results[i]))
