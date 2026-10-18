import os
import numpy as np
import csv
import openpyxl
from ss1 import ssvepDetect

if __name__ == '__main__':
    base_dir = r'C:/Users/Administrator/Desktop/v4'
    data_dir = os.path.join(base_dir, 'Data')
    excel_path = os.path.join(base_dir, 'result', '2025BCI-00291N-result.xlsx')

    srate = 250
    freqs = [8, 9, 10, 11, 12, 13, 14, 15]

    task_config = {
        'Task1': {'dataLen': 3, 'points': 750},
        'Task2': {'dataLen': 1, 'points': 250},
    }

    # S1-S12 对应 Excel 列 B-M
    col_map = {
        'S1': 'B', 'S2': 'C', 'S3': 'D', 'S4': 'E',
        'S5': 'F', 'S6': 'G', 'S7': 'H', 'S8': 'I',
        'S9': 'J', 'S10': 'K', 'S11': 'L', 'S12': 'M',
    }

    # 打开 Excel 模板
    wb = openpyxl.load_workbook(excel_path)
    ws = wb.active

    # 累计统计
    total_correct = 0
    total_trials = 0

    for task in ['Task1', 'Task2']:
        config = task_config[task]
        dataLen = config['dataLen']
        points = config['points']

        sd = ssvepDetect(srate, freqs, dataLen)

        task_dir = os.path.join(data_dir, task)
        task_correct = 0
        task_trials = 0

        for fname in sorted(os.listdir(task_dir)):
            if not fname.endswith('.csv'):
                continue

            stem = fname.replace('.csv', '')
            datapath = os.path.join(task_dir, fname)
            col_letter = col_map[stem]

            # 读取 CSV
            data = []
            with open(datapath, mode='r') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    data.append([float(v) for v in row])
            data = np.array(data, dtype=np.float64)

            # 从数据行数推算 trial 数，必须为 48
            n_trials = data.shape[0] // points
            if n_trials != 48:
                raise ValueError(
                    f'{task}/{stem}: 期望 48 trials，实际 {n_trials} trials '
                    f'(data rows={data.shape[0]}, points={points})'
                )

            correct = 0
            for i in range(n_trials):
                epoch = data[i * points:(i + 1) * points, :6].T
                res = sd.detect(epoch)
                true_label = int(data[i * points, -1])

                if res == true_label:
                    correct += 1

                # 写入 Excel
                ws[f'{col_letter}{7 + i}'] = res

            acc = correct / n_trials
            task_correct += correct
            task_trials += n_trials
            total_correct += correct
            total_trials += n_trials

            print(f'{task}/{stem}  {correct:2d}/{n_trials}  acc={acc:.4f}  (dataLen={dataLen}s)')

        task_acc = task_correct / task_trials
        print(f'  --- {task} 汇总: {task_correct}/{task_trials}  acc={task_acc:.4f} ---\n')

    overall_acc = total_correct / total_trials
    print(f'{"=" * 50}')
    print(f'总体: {total_correct}/{total_trials}  acc={overall_acc:.4f}')
    print(f'{"=" * 50}')

    wb.save(excel_path)
    print(f'\nExcel 已保存至 {excel_path}')
