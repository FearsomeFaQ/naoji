import os
import numpy as np
import csv
import openpyxl
from ss1 import ssvepDetect

if __name__ == '__main__':
    # 路径配置
    base_dir = r'C:/Users/Administrator/Desktop/v4'
    data_dir = os.path.join(base_dir, 'Data')
    excel_path = os.path.join(base_dir, 'result', '2025BCI-00291N-result.xlsx')

    # 实验参数
    srate = 250
    freqs = [8, 9, 10, 11, 12, 13, 14, 15]

    # Task1: 3 秒数据（750 点）
    # Task2: 1 秒数据（250 点）
    task_config = {
        'Task1': {'dataLen': 3, 'points': 750},
        'Task2': {'dataLen': 1, 'points': 250},
    }

    # 打开 Excel 模板
    wb = openpyxl.load_workbook(excel_path)
    ws = wb.active

    # S1-S12 对应 Excel 列 B-M
    col_map = {
        'S1': 'B', 'S2': 'C', 'S3': 'D', 'S4': 'E',
        'S5': 'F', 'S6': 'G', 'S7': 'H', 'S8': 'I',
        'S9': 'J', 'S10': 'K', 'S11': 'L', 'S12': 'M',
    }

    # 遍历 Task1、Task2
    for task in ['Task1', 'Task2']:
        config = task_config[task]
        dataLen = config['dataLen']
        points = config['points']

        # 为每个 Task 创建独立的检测器（模板长度不同）
        sd = ssvepDetect(srate, freqs, dataLen)

        task_dir = os.path.join(data_dir, task)
        for fname in sorted(os.listdir(task_dir)):
            if not fname.endswith('.csv'):
                continue

            stem = fname.replace('.csv', '')  # e.g. "S1"
            datapath = os.path.join(task_dir, fname)
            col_letter = col_map[stem]

            # 读取 CSV
            data = []
            with open(datapath, mode='r') as f:
                reader = csv.reader(f)
                next(reader)  # 跳过表头
                for row in reader:
                    data.append([float(v) for v in row])
            data = np.array(data, dtype=np.float64)

            # 逐片段检测并写入 Excel
            for i in range(48):
                epoch = data[i * points:(i + 1) * points, :6].T
                res = sd.detect(epoch)
                # 写入对应单元格（行7起始对应 trial 0）
                ws[f'{col_letter}{7 + i}'] = res

            print(f'{task}/{stem} 完成 (dataLen={dataLen}s)')

    wb.save(excel_path)
    print(f'\n结果已保存至 {excel_path}')
