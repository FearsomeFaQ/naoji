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


def test_accuracy_multi_offset(data_array, dataLen, srate, freqs,
                               n_harmonics, n_components, use_filterbank,
                               offsets, use_correction=True,
                               n_filter_bands=8, filter_order=4):
    """
    用多个偏移截取 4s 数据，综合评估准确率。
    data_array: 原始 4s 数据
    offsets: 截取偏移列表，如 [0, 125, 250]
    返回: (correct, total)
    """
    orig_points = 4 * srate
    test_points = dataLen * srate

    sd = ssvepDetect(
        srate, freqs, dataLen,
        n_harmonics=n_harmonics,
        n_components=n_components,
        use_filterbank=use_filterbank,
        use_correction=use_correction,
        n_filter_bands=n_filter_bands,
        filter_order=filter_order,
    )

    n_trials = data_array.shape[0] // orig_points
    correct = 0
    total = 0

    for i in range(n_trials):
        base = i * orig_points
        true_label = int(data_array[base, -1])
        for offset in offsets:
            start = base + offset
            epoch = data_array[start:start + test_points, :6].T
            pred = sd.detect(epoch)
            if pred == true_label:
                correct += 1
            total += 1

    return correct, total


def main():
    base = r'C:/Users/Administrator/Desktop/v4/ExampleData'
    d1 = load_data(f'{base}/D1.csv')
    d2 = load_data(f'{base}/D2.csv')

    srate = 250
    freqs = [8, 9, 10, 11, 12, 13, 14, 15]

    # ================================================================
    # Task1 (dataLen=3s) 深度扫描
    # 用 6 个偏移模拟多 trial 截取，更可靠地评估泛化能力
    # ================================================================
    offsets_3s = [0, 50, 100, 150, 200, 250]  # 6 个偏移，均匀覆盖 4s

    print("=" * 60)
    print("Task1 (dataLen=3s) 深度参数扫描")
    print(f"验证方式: D1+D2 各 trial × {len(offsets_3s)} 偏移 = {48 * len(offsets_3s) * 2} 次测试")
    print("=" * 60)

    results = []

    # 阶段 1: 基础参数 + 校正 + 谐波×分量
    print("\n--- 阶段 1: 谐波×分量×滤波×校正 ---")
    for nh, nc, fb, corr in product([2, 3, 4, 5], [2, 3, 4], [True, False], [True, False]):
        if not fb and not corr:
            continue
        c1, t1 = test_accuracy_multi_offset(d1, 3, srate, freqs, nh, nc, fb,
                                             offsets_3s, use_correction=corr)
        c2, t2 = test_accuracy_multi_offset(d2, 3, srate, freqs, nh, nc, fb,
                                             offsets_3s, use_correction=corr)
        acc = (c1 + c2) / (t1 + t2)
        cfg = {'nh': nh, 'nc': nc, 'fb': fb, 'corr': corr, 'n_fb': 8, 'f_order': 4}
        results.append((acc, cfg))
        marker = " <--" if acc > 0.90 else ""
        print(f"  nh={nh} nc={nc} fb={fb} corr={corr} => D1={c1}/{t1} D2={c2}/{t2} acc={acc:.4f}{marker}")

    # 阶段 2: 基于最佳基础参数，微调滤波子带和阶数
    results.sort(key=lambda x: x[0], reverse=True)
    best_base = results[0][1]
    best_base_acc = results[0][0]
    print(f"\n--- 阶段 2: 滤波微调 (基于 nh={best_base['nh']} nc={best_base['nc']} fb={best_base['fb']} corr={best_base['corr']}) ---")

    for n_fb, f_order in product([4, 6, 8, 10], [2, 3, 4, 6]):
        c1, t1 = test_accuracy_multi_offset(d1, 3, srate, freqs,
                                             best_base['nh'], best_base['nc'], best_base['fb'],
                                             offsets_3s, use_correction=best_base['corr'],
                                             n_filter_bands=n_fb, filter_order=f_order)
        c2, t2 = test_accuracy_multi_offset(d2, 3, srate, freqs,
                                             best_base['nh'], best_base['nc'], best_base['fb'],
                                             offsets_3s, use_correction=best_base['corr'],
                                             n_filter_bands=n_fb, filter_order=f_order)
        acc = (c1 + c2) / (t1 + t2)
        cfg = {'nh': best_base['nh'], 'nc': best_base['nc'], 'fb': best_base['fb'],
               'corr': best_base['corr'], 'n_fb': n_fb, 'f_order': f_order}
        results.append((acc, cfg))
        marker = " <--" if acc > best_base_acc else ""
        print(f"  n_bands={n_fb} order={f_order} => acc={acc:.4f}{marker}")
        best_base_acc = results[0][0]

    # 排序输出
    results.sort(key=lambda x: x[0], reverse=True)

    print(f"\n{'=' * 60}")
    print(f"Task1 前 10 名:")
    for i, (acc, cfg) in enumerate(results[:10]):
        print(f"  {i + 1:2d}. acc={acc:.4f}  {cfg}")

    print(f"\n最佳参数: {results[0][1]}")
    print(f"最佳准确率: {results[0][0]:.4f}")

    # 对比默认参数
    c1, t1 = test_accuracy_multi_offset(d1, 3, srate, freqs, 3, 3, True,
                                         offsets_3s, use_correction=True, n_filter_bands=8, filter_order=4)
    c2, t2 = test_accuracy_multi_offset(d2, 3, srate, freqs, 3, 3, True,
                                         offsets_3s, use_correction=True, n_filter_bands=8, filter_order=4)
    default_acc = (c1 + c2) / (t1 + t2)
    improvement = (results[0][0] - default_acc) * 100
    print(f"\n默认参数 acc={default_acc:.4f}, 最佳提升: {improvement:+.1f}%")
    print(f"\n{'=' * 60}")
    print("校准完成")
    print("=" * 60)


if __name__ == '__main__':
    main()
