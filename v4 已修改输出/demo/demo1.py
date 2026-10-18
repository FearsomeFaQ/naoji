import numpy as np
import csv
from ss1 import ssvepDetect

if __name__ == '__main__':
    # 实验数据路径
    datapath = r'C:/Users/Administrator/Desktop/v4/ExampleData/D1.csv'

    # 实验参数
    srate = 250 # 采样率250Hz
    dataLen = 4 # example数据长度为4秒

    # 实例化ssvep检测器
    # 输入参数：
    # srate: 采样率250Hz
    # 实验刺激频率：[8,9,10,11,12,13,14,15] 不要修改
    # dataLen：待分析数据信号片段的长度
    sd = ssvepDetect(srate,[8,9,10,11,12,13,14,15],dataLen)

    # 读取数据
    data = [] # 用来存储所有原始数据

    # 打开CSV文件，读取数据
    with open(datapath, mode='r') as file:
        csv_reader = csv.reader(file)
        # 跳过第一行表头
        next(csv_reader)

        for row in csv_reader:
            # csv以字符串形式存储，需要转换成浮点型
            rowvalue = [float(_) for _ in row]
            # 所有数据整理后存入data列表中
            data.append(rowvalue)

    # 将列表型转换成np.array型，便于后续处理
    data = np.array(data,dtype=np.float64)

    points = dataLen * srate
    results = []
    stimIDs = []
    corr = []

    # 每个数据中都有48个片段
    for i in range(48):
        epoch = data[i*points:(i+1)*points,:6] # 把这一段的6个通道信号片段取出
        epoch = epoch.transpose() # 以行来组织，每一行是一个通道的数据
        res = sd.detect(epoch) # 识别，得到的结果res取值范围是0-7
        results.append(res)
        # 如果这是示例数据，则能够得到真值
        stim = int(data[i*points,-1])
        stimIDs.append(stim)

        if res == stim:
            correct = 1
        else:
            correct = 0

        corr.append(correct)

    print("正确率： %.2f"%(sum(corr)/48))

    # results里面包含了所有的预测值，应当按顺序填写到result.csv中，并将结果反馈至组委会
    for i in range(48):
        print("task%d预测值：%d"%(i,results[i]))

