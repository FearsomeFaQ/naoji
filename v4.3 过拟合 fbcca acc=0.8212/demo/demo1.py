import os
import numpy as np
import csv
from ss1 import ssvepDetect

if __name__ == '__main__':
    base_dir = r'C:/Users/Administrator/Desktop/v4/ExampleData'
    srate = 250
    freqs = [8, 9, 10, 11, 12, 13, 14, 15]

    config = {
        'Task1': {'dataLen': 3, 'points': 750},
        'Task2': {'dataLen': 1, 'points': 250},
    }

    total_correct = 0
    total_trials = 0

    for task_name, cfg in config.items():
        task_dir = os.path.join(base_dir, task_name)
        dataLen = cfg['dataLen']
        points = cfg['points']

        sd = ssvepDetect(srate, freqs, dataLen)

        task_correct = 0
        task_trials = 0

        for fname in sorted(os.listdir(task_dir)):
            if not fname.endswith('.csv'):
                continue

            stem = fname.replace('.csv', '')
            datapath = os.path.join(task_dir, fname)

            data = []
            with open(datapath, mode='r') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    data.append([float(v) for v in row])
            data = np.array(data, dtype=np.float64)

            n_trials = data.shape[0] // points

            correct = 0
            for i in range(n_trials):
                epoch = data[i * points:(i + 1) * points, :6].T
                res = sd.detect(epoch)
                true_label = int(data[i * points, -1])
                if res == true_label:
                    correct += 1

            acc = correct / n_trials
            task_correct += correct
            task_trials += n_trials
            total_correct += correct
            total_trials += n_trials

            print(f'[Example] {task_name}/{stem}  {correct:3d}/{n_trials}  acc={acc:.4f}  (dataLen={dataLen}s)')

        task_acc = task_correct / task_trials
        print(f'  --- [Example] {task_name} 汇总: {task_correct}/{task_trials}  acc={task_acc:.4f} ---\n')

    overall_acc = total_correct / total_trials
    print(f'{"=" * 50}')
    print(f'总体: {total_correct}/{total_trials}  acc={overall_acc:.4f}')
    print(f'{"=" * 50}')
