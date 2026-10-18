import numpy as np
import csv
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


def test_accuracy(data_array, dataLen, srate, freqs,
                  n_harmonics, n_components, use_filterbank,
                  chans=None, use_correction=True,
                  n_filter_bands=8, filter_order=4):
    orig_points = 4 * srate
    test_points = dataLen * srate

    sd = ssvepDetect(
        srate, freqs, dataLen,
        n_harmonics=n_harmonics,
        n_components=n_components,
        use_filterbank=use_filterbank,
        chans=chans,
        use_correction=use_correction,
        n_filter_bands=n_filter_bands,
        filter_order=filter_order,
    )

    n_trials = 48
    correct = 0
    for i in range(n_trials):
        start = i * orig_points
        epoch = data_array[start:start + test_points, :6].T
        pred = sd.detect(epoch)
        true_label = int(data_array[start, -1])
        if pred == true_label:
            correct += 1
    return correct / n_trials


def main():
    base = r'C:/Users/Administrator/Desktop/v4/ExampleData'
    files = {'D1': f'{base}/D1.csv', 'D2': f'{base}/D2.csv'}

    srate = 250
    freqs = [8, 9, 10, 11, 12, 13, 14, 15]

    # 预加载数据
    all_data = {}
    for name, path in files.items():
        all_data[name] = load_data(path)

    # ====== dataLen=3 (Task1) — 快速确认 ======
    print("=" * 60)
    print("dataLen=3s (Task1) — 基础参数扫描")
    print("=" * 60)
    best_3 = (-1, None)
    for nh, nc, fb in product([2, 3, 4, 5], [2, 3], [True, False]):
        accs = [test_accuracy(all_data[n], 3, srate, freqs, nh, nc, fb) for n in ['D1', 'D2']]
        avg = sum(accs) / 2
        print(f"  nh={nh} nc={nc} fb={fb} => {avg:.4f}")
        if avg > best_3[0]:
            best_3 = (avg, {'nh': nh, 'nc': nc, 'fb': fb})
    print(f"dataLen=3 最佳: {best_3[1]}, avg={best_3[0]:.4f}")

    # ====== dataLen=1 (Task2) — 深度测试 ======
    print("\n" + "=" * 60)
    print("dataLen=1s (Task2) — 深度参数扫描")
    print("=" * 60)

    best_1 = (-1, None)
    results_1 = []

    # 测试 1: 基础参数 + 校正开关
    print("\n--- 基础参数 + 校正开关 ---")
    for nh, nc, fb, corr in product([2, 3, 4, 5], [2, 3], [True, False], [True, False]):
        if not fb and not corr:
            continue  # 无FB时corr无效
        accs = [test_accuracy(all_data[n], 1, srate, freqs, nh, nc, fb,
                              use_correction=corr) for n in ['D1', 'D2']]
        avg = sum(accs) / 2
        print(f"  nh={nh} nc={nc} fb={fb} corr={corr} => {avg:.4f}")
        results_1.append((avg, {'nh': nh, 'nc': nc, 'fb': fb, 'corr': corr,
                                'n_fb': 8, 'f_order': 4}))
        if avg > best_1[0]:
            best_1 = (avg, results_1[-1][1])

    # 测试 2: 减少滤波子带 (n_filter_bands=4,6) + 最佳基础参数
    print("\n--- 减少滤波子带 (corr=False, nh=5) ---")
    for n_fb, f_order in product([4, 6], [2, 3, 4]):
        accs = [test_accuracy(all_data[n], 1, srate, freqs, 5, 2, True,
                              use_correction=False, n_filter_bands=n_fb,
                              filter_order=f_order) for n in ['D1', 'D2']]
        avg = sum(accs) / 2
        print(f"  n_bands={n_fb} order={f_order} => {avg:.4f}")
        results_1.append((avg, {'nh': 5, 'nc': 2, 'fb': True, 'corr': False,
                                'n_fb': n_fb, 'f_order': f_order}))
        if avg > best_1[0]:
            best_1 = (avg, results_1[-1][1])

    # 排序
    results_1.sort(key=lambda x: x[0], reverse=True)

    print(f"\n{'='*60}")
    print(f"dataLen=3 最佳: {best_3[1]}, avg={best_3[0]:.4f}")
    print(f"dataLen=1 最佳: {best_1[1]}, avg={best_1[0]:.4f}")
    print(f"\ndataLen=1 前10名:")
    for i, (acc, cfg) in enumerate(results_1[:10]):
        print(f"  {i+1}. avg={acc:.4f}  {cfg}")

    print(f"\n{'='*60}")
    print("校准完成")
    print("=" * 60)


if __name__ == '__main__':
    main()
