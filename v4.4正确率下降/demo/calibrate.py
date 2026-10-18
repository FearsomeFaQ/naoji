import numpy as np
import csv
import os
from itertools import product
from ss1 import ssvepDetect


def load_data(datapath):
    data = []
    with open(datapath, mode='r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            data.append([float(v) for v in row])
    return np.array(data, dtype=np.float64)


def test_accuracy(data_dir, dataLen, srate, freqs,
                  n_harmonics, n_components, use_filterbank,
                  use_correction=True, n_filter_bands=8, filter_order=4):
    """
    直接读取目录下所有 CSV，按 dataLen 切分 trial 并评估准确率。
    返回: (correct, total)
    """
    points = dataLen * srate
    sd = ssvepDetect(
        srate, freqs, dataLen,
        n_harmonics=n_harmonics,
        n_components=n_components,
        use_filterbank=use_filterbank,
        use_correction=use_correction,
        n_filter_bands=n_filter_bands,
        filter_order=filter_order,
    )

    correct = 0
    total = 0

    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith('.csv'):
            continue
        data = load_data(os.path.join(data_dir, fname))
        n_trials = data.shape[0] // points

        for i in range(n_trials):
            epoch = data[i * points:(i + 1) * points, :6].T
            true_label = int(data[i * points, -1])
            pred = sd.detect(epoch)
            if pred == true_label:
                correct += 1
            total += 1

    return correct, total


def run_grid_search(task_name, data_dir, dataLen, srate, freqs):
    """对单个 Task 执行两阶段网格搜索"""
    print("=" * 60)
    print(f"{task_name} (dataLen={dataLen}s) 参数扫描")
    print("=" * 60)

    results = []

    # 阶段 1: 谐波×分量×滤波×校正
    print("\n--- 阶段 1: 谐波×分量×滤波×校正 ---")
    for nh, nc, fb, corr in product([2, 3, 4, 5], [2, 3, 4], [True, False], [True, False]):
        if not fb and not corr:
            continue
        c, t = test_accuracy(data_dir, dataLen, srate, freqs, nh, nc, fb,
                             use_correction=corr)
        acc = c / t
        results.append((acc, {'nh': nh, 'nc': nc, 'fb': fb, 'corr': corr, 'n_fb': 8, 'f_order': 4}))
        marker = " <--" if acc > 0.85 else ""
        print(f"  nh={nh} nc={nc} fb={fb} corr={corr} => {c}/{t} acc={acc:.4f}{marker}")

    # 阶段 2: 滤波微调
    results.sort(key=lambda x: x[0], reverse=True)
    best_base = results[0][1]
    print(f"\n--- 阶段 2: 滤波微调 (基于 nh={best_base['nh']} nc={best_base['nc']} fb={best_base['fb']} corr={best_base['corr']}) ---")

    for n_fb, f_order in product([4, 6, 8, 10], [2, 3, 4, 6]):
        c, t = test_accuracy(data_dir, dataLen, srate, freqs,
                             best_base['nh'], best_base['nc'], best_base['fb'],
                             use_correction=best_base['corr'],
                             n_filter_bands=n_fb, filter_order=f_order)
        acc = c / t
        cfg = {'nh': best_base['nh'], 'nc': best_base['nc'], 'fb': best_base['fb'],
               'corr': best_base['corr'], 'n_fb': n_fb, 'f_order': f_order}
        results.append((acc, cfg))
        best_in_stage1 = results[0][0]
        marker = " <--" if acc >= best_in_stage1 else ""
        print(f"  n_bands={n_fb} order={f_order} => acc={acc:.4f}{marker}")

    # 排序输出
    results.sort(key=lambda x: x[0], reverse=True)

    print(f"\n{'=' * 60}")
    print(f"{task_name} 前 10 名:")
    for i, (acc, cfg) in enumerate(results[:10]):
        print(f"  {i + 1:2d}. acc={acc:.4f}  {cfg}")

    print(f"\n最佳参数: {results[0][1]}")
    print(f"最佳准确率: {results[0][0]:.4f}")

    # 对比默认参数
    c, t = test_accuracy(data_dir, dataLen, srate, freqs, 3, 3, True,
                         use_correction=True, n_filter_bands=8, filter_order=4)
    default_acc = c / t
    improvement = (results[0][0] - default_acc) * 100
    print(f"\n默认参数 acc={default_acc:.4f}, 最佳提升: {improvement:+.1f}%")
    print(f"{'=' * 60}\n")

    return results[0]


def main():
    base = r'C:/Users/Administrator/Desktop/v4/ExampleData'
    srate = 250
    freqs = [8, 9, 10, 11, 12, 13, 14, 15]

    best_task1 = run_grid_search('Task1', os.path.join(base, 'Task1'), 3, srate, freqs)
    best_task2 = run_grid_search('Task2', os.path.join(base, 'Task2'), 1, srate, freqs)

    print("=" * 60)
    print("最终推荐参数:")
    print(f"  Task1 (dataLen=3): {best_task1[1]}")
    print(f"  Task2 (dataLen=1): {best_task2[1]}")
    print("=" * 60)


if __name__ == '__main__':
    main()
