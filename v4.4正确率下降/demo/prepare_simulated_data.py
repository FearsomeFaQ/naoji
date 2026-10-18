import numpy as np
import csv
import os
import random


def load_data(datapath):
    data = []
    with open(datapath, mode='r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            data.append([float(v) for v in row])
    return np.array(data, dtype=np.float64)


def extract_segments(data_array, window_samples, offsets, srate=250):
    orig_points = 4 * srate
    n_trials = data_array.shape[0] // orig_points
    segments = []
    for i in range(n_trials):
        start = i * orig_points
        stim_id = int(data_array[start, -1])
        for offset in offsets:
            seg = data_array[start + offset: start + offset + window_samples, :].copy()
            segments.append((seg, stim_id))
    return segments


def write_subject_csv(segments, output_path):
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['CP3', 'CPZ', 'CP4', 'PO3', 'POZ', 'PO4', 'taskID', 'stimID'])
        for seg_data, stim_id in segments:
            for row in seg_data:
                writer.writerow(list(row))


def generate_split(combined, window_samples, offsets, out_dir, name_pattern,
                   n_subjects=6, seed=42):
    """通用：提取片段、打乱、分人、写入 CSV。"""
    segs = extract_segments(combined, window_samples, offsets)
    random.seed(seed)
    random.shuffle(segs)
    t_per = len(segs) // n_subjects
    os.makedirs(out_dir, exist_ok=True)

    info = []
    for subj in range(n_subjects):
        subset = segs[subj * t_per: (subj + 1) * t_per]
        fname = name_pattern.format(subj + 1)
        path = os.path.join(out_dir, fname)
        write_subject_csv(subset, path)
        info.append((path, len(subset)))
    return info


def main():
    base = r'C:/Users/Administrator/Desktop/v4'
    example_dir = os.path.join(base, 'ExampleData')

    # 加载 D1 + D2
    d1 = load_data(os.path.join(example_dir, 'D1.csv'))
    d2 = load_data(os.path.join(example_dir, 'D2.csv'))
    combined = np.vstack([d1, d2])
    print(f'D1+D2 整合: {combined.shape[0]} 行 ({combined.shape[0] // 1000} trials)')

    # ============================================================
    # Data/ — 标准 48 trial/人
    # ============================================================
    print("\n" + "=" * 60)
    print("Data/ 标准版 (48 trial/人)")
    print("=" * 60)

    # Task1: 3 偏移 [0, 125, 250] → 288 段 / 6 = 48
    info = generate_split(combined, 750, [0, 125, 250],
                          os.path.join(base, 'Data', 'Task1'),
                          'S{}.csv', n_subjects=6, seed=42)
    for p, n in info:
        print(f'  {p}  — {n} trials')

    # Task2: 3 偏移 [0, 375, 750] → 288 段 / 6 = 48
    info = generate_split(combined, 250, [0, 375, 750],
                          os.path.join(base, 'Data', 'Task2'),
                          'S{}.csv', n_subjects=6, seed=42)
    for p, n in info:
        new_name = p.replace('S', 'S').replace('.csv', '')
        # 文件名是 S{idx}.csv，需要改成 S{idx+6}
        pass

    # 重命名 Task2: S1~S6 → S7~S12
    t2_dir = os.path.join(base, 'Data', 'Task2')
    for i in range(6, 0, -1):
        old = os.path.join(t2_dir, f'S{i}.csv')
        new = os.path.join(t2_dir, f'S{i + 6}.csv')
        if os.path.exists(old):
            os.rename(old, new)
            print(f'  重命名: S{i}.csv → S{i + 6}.csv')

    # ============================================================
    # ExampleData/ — 多 trial 版本
    #   Task1 (3s): 6 偏移 stride≈50 → 576 段 / 6 = 96 trial/人
    #   Task2 (1s): 6 偏移 stride≈150 → 576 段 / 6 = 96 trial/人
    # ============================================================
    print("\n" + "=" * 60)
    print("ExampleData/ 多 trial 版 (96 trial/人)")
    print("=" * 60)

    info = generate_split(combined, 750,
                          [0, 50, 100, 150, 200, 250],
                          os.path.join(example_dir, 'Task1_More'),
                          'S{}.csv', n_subjects=6, seed=99)
    for p, n in info:
        print(f'  {p}  — {n} trials')

    info = generate_split(combined, 250,
                          [0, 150, 300, 450, 600, 750],
                          os.path.join(example_dir, 'Task2_More'),
                          'S{}.csv', n_subjects=6, seed=99)
    for p, n in info:
        new_name = p  # keep as is, these are S1~S6
        print(f'  {p}  — {n} trials')

    # 重命名 ExampleData Task2_More: S1~S6 → S7~S12
    t2_more_dir = os.path.join(example_dir, 'Task2_More')
    for i in range(6, 0, -1):
        old = os.path.join(t2_more_dir, f'S{i}.csv')
        new = os.path.join(t2_more_dir, f'S{i + 6}.csv')
        if os.path.exists(old):
            os.rename(old, new)
            print(f'  重命名: S{i}.csv → S{i + 6}.csv')

    print("\n完成！")


if __name__ == '__main__':
    main()
