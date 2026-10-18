"""
SSVEP 比赛训练脚本 —— 小样本跨被试泛化方案
==============================================
Task1（3s）：纯 FB-eCCA（不训练模型）
Task2（1s）：FB-eCCA + FB-eTRCA 集成（0.85/0.15 固定融合）

策略重心：泛化能力 > 训练集准确率
"""

import numpy as np
import pickle
from scipy import signal
from scipy.linalg import eigh
from sklearn.cross_decomposition import CCA

# ============================================================
# 全局常量
# ============================================================
SRATE = 250
FREQS = [8, 9, 10, 11, 12, 13, 14, 15]                  # 刺激频率 (Hz)
N_CLASSES = len(FREQS)                                     # 8
N_CHANS = 6                                                # 导联数
FILTERBANK_CUTS = [(6, 70), (14, 70), (22, 70), (30, 70), (38, 70)]  # 5 个子带
N_FILTERBANKS = len(FILTERBANK_CUTS)                       # 5
FB_LAMBDA = 0.2                                            # LW shrinkage 系数
CST_N_SHIFTS = 2                                           # 增强副本数
CST_SHIFT_RANGE = (15, 50)                                 # 移位范围（采样点）
FUSION_W_ECCA = 0.85                                       # eCCA 融合权重
FUSION_W_ETRCA = 0.15                                      # eTRCA 融合权重

# 数据路径
DATA_DIR = r"C:\Users\Administrator\Desktop\v3\ExampleData"
OUTPUT_DIR = r"C:\Users\Administrator\Desktop\v3\demo"


# ============================================================
# 1. load_trials_by_taskid —— 按 taskID 切分 trial
# ============================================================
def load_trials_by_taskid(csv_path, target_len, start_offset=0):
    """
    读取 CSV 文件，根据 taskID 变化自动切分成独立 trial。

    参数
    ----
    csv_path : str
        CSV 文件路径（含表头）。
    target_len : int
        每个 trial 截取的采样点数（750 表示 3 秒，250 表示 1 秒）。
    start_offset : int
        从每个 trial 开头跳过的采样点数（SSVEP 前 100-200ms 可能不稳定）。

    返回
    ----
    trials : list of (eeg, stimID)
        eeg.shape == (6, target_len)，stimID ∈ {0,...,7}。
    """
    # 读取原始数据，跳过表头
    with open(csv_path, "r") as f:
        lines = f.readlines()

    header = lines[0].strip().split(",")
    n_cols = len(header)  # 应为 8

    raw = []
    for line in lines[1:]:
        parts = line.strip().split(",")
        if len(parts) < n_cols:
            continue
        raw.append([float(v) for v in parts])
    raw = np.array(raw, dtype=np.float64)  # (total_samples, 8)

    # 按 taskID 变化点切分
    task_ids = raw[:, 6]
    change_points = np.where(np.diff(task_ids) != 0)[0] + 1
    segment_starts = np.concatenate([[0], change_points])
    segment_ends = np.concatenate([change_points, [len(raw)]])

    trials = []
    for seg_start, seg_end in zip(segment_starts, segment_ends):
        seg_len = seg_end - seg_start
        if seg_len < start_offset + target_len:
            continue  # 跳过长度不足的片段

        # 截取指定窗口
        win_start = seg_start + start_offset
        win_end = win_start + target_len
        eeg = raw[win_start:win_end, :6].T.copy()  # (6, target_len)

        # 标签：取该片段的众数 stimID（或第一个值）
        stim_ids_in_seg = raw[seg_start:seg_end, 7].astype(int)
        stim_id = int(np.bincount(stim_ids_in_seg).argmax())

        trials.append((eeg, stim_id))

    return trials


# ============================================================
# 2. preprocess_trial —— 预处理
# ============================================================
def preprocess_trial(trial, srate=SRATE):
    """
    对单个 trial 进行预处理。

    步骤：50Hz 陷波 → 6-40Hz 带通 → detrend → CAR → z-score

    参数
    ----
    trial : ndarray, shape (6, samples)
        原始 EEG 片段。
    srate : int
        采样率。

    返回
    ----
    out : ndarray, shape (6, samples)
        预处理后的 EEG。
    """
    data = trial.copy()

    # --- 50 Hz 陷波 (Q=30) ---
    b_notch, a_notch = signal.iirnotch(50, 30, fs=srate)
    data = signal.filtfilt(b_notch, a_notch, data, axis=1)

    # --- 6-40 Hz 带通 (4 阶 Butterworth) ---
    nyq = srate / 2.0
    b_bp, a_bp = signal.butter(4, [6.0 / nyq, 40.0 / nyq], btype="bandpass")
    data = signal.filtfilt(b_bp, a_bp, data, axis=1)

    # --- 去趋势 ---
    data = signal.detrend(data, axis=1)

    # --- CAR（共平均参考） ---
    data = data - np.mean(data, axis=0, keepdims=True)

    # --- z-score 归一化（按通道） ---
    ch_mean = np.mean(data, axis=1, keepdims=True)
    ch_std = np.std(data, axis=1, keepdims=True)
    data = (data - ch_mean) / (ch_std + 1e-6)

    return data


# ============================================================
# 3. cyclic_shift_augment —— 循环移位增强
# ============================================================
def cyclic_shift_augment(trial, n_shifts=CST_N_SHIFTS,
                         shift_range=CST_SHIFT_RANGE):
    """
    通过循环移位生成增强副本，模拟 phase jitter。

    参数
    ----
    trial : ndarray, shape (6, samples)
        原始 trial。
    n_shifts : int
        生成的增强副本数量。
    shift_range : tuple (lo, hi)
        随机移位范围（采样点）。

    返回
    ----
    augmented : list of ndarray
        [original, aug1, aug2, ...]，每个元素 shape (6, samples)。
    """
    rng = np.random.RandomState(42)  # 固定种子保证可复现
    lo, hi = shift_range
    samples = trial.shape[1]
    augmented = [trial]

    for _ in range(n_shifts):
        shift = rng.randint(lo, hi + 1)
        # 随机方向
        if rng.rand() > 0.5:
            shift = -shift
        aug = np.roll(trial, shift, axis=1)
        augmented.append(aug)

    return augmented


# ============================================================
# 4. train_fb_etrca —— 训练滤波器组 eTRCA
# ============================================================
def train_fb_etrca(train_data_by_freq):
    """
    训练 FilterBank Ensemble TRCA 模型。

    对 5 个子带各自独立训练 TRCA：
      - Ledoit-Wolf 风格 shrinkage 处理 Q 矩阵
      - 广义特征值分解获取空间滤波器
      - 模板符号对齐防止平均抵消

    参数
    ----
    train_data_by_freq : dict
        按频率分组的训练数据，{class_id: [trial_1, trial_2, ...]}。
        每个 trial shape = (6, samples)。

    返回
    ----
    model : dict
        {
            'filterbanks': [(b_band1, a_band1), ...],    # 5 个滤波器系数
            'bands': [
                {
                    'W_ensemble': ndarray (6, 8),
                    'templates': {class_id: ndarray (8, samples)},
                },
                ...
            ],
            'fb_weights': ndarray (5,),                   # 子带权重
        }
    """
    n_samples = next(iter(train_data_by_freq.values()))[0].shape[1]
    nyq = SRATE / 2.0

    # ----- 预构建滤波器 -----
    filterbanks = []
    for low_cut, high_cut in FILTERBANK_CUTS:
        b, a = signal.butter(4, [low_cut / nyq, high_cut / nyq], btype="bandpass")
        filterbanks.append((b, a))

    # ----- FB 权重：w_m = m^(-1.25) + 0.25 -----
    raw_w = np.array([m ** (-1.25) + 0.25 for m in range(1, N_FILTERBANKS + 1)])
    fb_weights = raw_w / raw_w.sum()

    bands = []

    for band_idx, (b_band, a_band) in enumerate(filterbanks):
        # 对该子带滤波所有 trial
        filtered_data = {}
        for cls_id, trials in train_data_by_freq.items():
            filtered_data[cls_id] = [
                signal.filtfilt(b_band, a_band, t, axis=1) for t in trials
            ]

        # 对每个频率独立训练 TRCA 空间滤波器
        w_list = []  # 存放各频率的空间滤波器

        for cls_id in sorted(filtered_data.keys()):
            trials_f = filtered_data[cls_id]  # list of (6, samples)
            N = len(trials_f)
            C = N_CHANS  # 6
            S = n_samples

            # ----- 计算 Q_f（池化协方差） -----
            Q_f = np.zeros((C, C))
            for i in range(N):
                X_i = trials_f[i]  # (6, samples)
                Q_f += X_i @ X_i.T

            # ----- 计算 S_f（试次间协方差） -----
            # S = (sum_i X_i)(sum_j X_j)^T - sum_i X_i X_i^T = N^2 X̄ X̄^T - Q_f
            sum_all = np.zeros((C, S))
            for i in range(N):
                sum_all += trials_f[i]
            X_mean = sum_all / N  # (6, samples)
            S_f = N * N * (X_mean @ X_mean.T) - Q_f

            # ----- Ledoit-Wolf shrinkage on Q_f -----
            trace_Q = np.trace(Q_f)
            Q_shrink = (1.0 - FB_LAMBDA) * Q_f + FB_LAMBDA * (trace_Q / C) * np.eye(C)

            # 对称化（消除数值误差）
            S_f = (S_f + S_f.T) / 2.0
            Q_shrink = (Q_shrink + Q_shrink.T) / 2.0

            # ----- 广义特征值分解 -----
            # S_f w = λ Q_shrink w
            eigvals, eigvecs = eigh(S_f, Q_shrink)
            # 取最大特征值对应的特征向量
            w_f = eigvecs[:, -1]  # (6,)
            w_list.append(w_f)

        # ----- 构建集成滤波器 W_ensemble (6, 8) -----
        W_ensemble = np.column_stack(w_list)  # (6, 8)
        # 对每个滤波器做 L2 归一化（可选，增强数值稳定性）
        for col in range(N_CLASSES):
            norm = np.linalg.norm(W_ensemble[:, col])
            if norm > 1e-10:
                W_ensemble[:, col] /= norm

        # ----- 构建模板（带符号对齐） -----
        templates = {}
        for cls_id in sorted(filtered_data.keys()):
            trials_f = filtered_data[cls_id]
            # 先用 W_ensemble 滤波
            proj_trials = [W_ensemble.T @ t for t in trials_f]  # 每个 (8, samples)

            # 计算初始均值
            mean_template = np.mean(proj_trials, axis=0)  # (8, samples)

            # 符号对齐：若 corr(trial, mean) < 0 则翻转
            aligned = []
            for pt in proj_trials:
                r = np.corrcoef(pt.flatten(), mean_template.flatten())[0, 1]
                if r < 0:
                    aligned.append(-pt)
                else:
                    aligned.append(pt)

            template = np.mean(aligned, axis=0)  # (8, samples)
            templates[cls_id] = template

        bands.append({
            "W_ensemble": W_ensemble,
            "templates": templates,
        })

    model = {
        "filterbanks": filterbanks,
        "bands": bands,
        "fb_weights": fb_weights,
    }
    return model


# ============================================================
# 5. predict_fb_etrca —— FB-eTRCA 预测
# ============================================================
def predict_fb_etrca(trial, model):
    """
    使用 FB-eTRCA 模型对单个 trial 进行预测。

    处理流程：
      每子带滤波 → W_ensemble^T @ X → flatten 相关 → sign(r)*r²
      → 子带内 z-score 归一化 → FB 加权聚合

    参数
    ----
    trial : ndarray, shape (6, samples)
        预处理后的单试次 EEG。
    model : dict
        train_fb_etrca 返回的模型。

    返回
    ----
    scores : ndarray, shape (8,)
        8 个频率类别的得分。
    """
    all_scores = np.zeros((N_FILTERBANKS, N_CLASSES))

    for band_idx, band_info in enumerate(model["bands"]):
        b_band, a_band = model["filterbanks"][band_idx]
        W_ensemble = band_info["W_ensemble"]  # (6, 8)
        templates = band_info["templates"]     # {cls_id: (8, samples)}

        # 滤波当前子带
        filtered = signal.filtfilt(b_band, a_band, trial, axis=1)  # (6, samples)
        y = W_ensemble.T @ filtered  # (8, samples)

        # 与各频率模板计算 flatten 相关
        for cls_id in range(N_CLASSES):
            template = templates[cls_id]  # (8, samples)
            r = np.corrcoef(y.flatten(), template.flatten())[0, 1]
            # sign(r) * r^2
            all_scores[band_idx, cls_id] = np.sign(r) * (r ** 2)

        # 子带内 z-score 归一化
        band_mean = all_scores[band_idx].mean()
        band_std = all_scores[band_idx].std()
        all_scores[band_idx] = (all_scores[band_idx] - band_mean) / (band_std + 1e-6)

    # FB 加权聚合
    scores = np.dot(model["fb_weights"], all_scores)  # (8,)
    return scores


# ============================================================
# 6. build_fb_ecca_references —— 构建 FB-eCCA 参考信号
# ============================================================
def build_fb_ecca_references(train_data_by_freq, n_samples, srate=SRATE):
    """
    构建 FB-eCCA 的增强参考信号。

    每个频率的参考 = [理论 sin/cos 模板 (10, n_samples); 训练数据均值 (6, n_samples)]
    总维度 (16, n_samples)。对 5 个子带分别滤波。

    参数
    ----
    train_data_by_freq : dict
        {class_id: [trial, ...]}，trial shape (6, n_samples)。
    n_samples : int
        每个 trial 的采样点数。
    srate : int
        采样率。

    返回
    ----
    ecca_refs : dict
        {
            'refs_by_freq': {freq_id: ndarray (16, n_samples)},
            'filterbanks': [(b, a), ...],
            'fb_weights': ndarray (5,),
            'n_harmonics': 5,
            'n_components': 2,
        }
    """
    nyq = srate / 2.0
    t = np.arange(n_samples) / srate

    # ----- 构建滤波器 -----
    filterbanks = []
    for low_cut, high_cut in FILTERBANK_CUTS:
        b, a = signal.butter(4, [low_cut / nyq, high_cut / nyq], btype="bandpass")
        filterbanks.append((b, a))

    # ----- FB 权重 -----
    raw_w = np.array([m ** (-1.25) + 0.25 for m in range(1, N_FILTERBANKS + 1)])
    fb_weights = raw_w / raw_w.sum()

    # ----- 构建每个频率的增强参考 -----
    refs_by_freq = {}
    for cls_id in sorted(train_data_by_freq.keys()):
        freq = FREQS[cls_id]

        # 理论模板：5 次谐波 sin + cos
        harmonics = []
        for h in range(1, 6):
            harmonics.append(np.sin(2 * np.pi * freq * h * t))
            harmonics.append(np.cos(2 * np.pi * freq * h * t))
        Y_theory = np.vstack(harmonics)  # (10, n_samples)

        # 训练数据平均模板
        trials = train_data_by_freq[cls_id]
        X_avg = np.mean(trials, axis=0)  # (6, n_samples)

        # 拼接
        Y_ref = np.vstack([Y_theory, X_avg])  # (16, n_samples)
        refs_by_freq[cls_id] = Y_ref

    ecca_refs = {
        "refs_by_freq": refs_by_freq,
        "filterbanks": filterbanks,
        "fb_weights": fb_weights,
        "n_harmonics": 5,
        "n_components": 2,
    }
    return ecca_refs


# ============================================================
# 辅助：FB-eCCA 预测
# ============================================================
def predict_fb_ecca(trial, ecca_refs):
    """
    使用 FB-eCCA 对单个 trial 进行预测。

    对每个子带：带通滤波试次和参考 → CCA → 取首成分相关系数
    → 子带内 z-score 归一化 → FB 加权聚合。

    参数
    ----
    trial : ndarray, shape (6, samples)
        预处理后的单试次 EEG。
    ecca_refs : dict
        build_fb_ecca_references 的返回值。

    返回
    ----
    scores : ndarray, shape (8,)
    """
    n_comp = ecca_refs["n_components"]
    all_scores = np.zeros((N_FILTERBANKS, N_CLASSES))

    for band_idx, (b_band, a_band) in enumerate(ecca_refs["filterbanks"]):
        # 滤波试次
        filtered_trial = signal.filtfilt(b_band, a_band, trial, axis=1)  # (6, S)

        for cls_id in range(N_CLASSES):
            # 滤波参考
            Y_ref = ecca_refs["refs_by_freq"][cls_id]  # (16, S)
            Y_filt = signal.filtfilt(b_band, a_band, Y_ref, axis=1)  # (16, S)

            # CCA：X = 试次^T (S, 6)，Y = 参考^T (S, 16)
            X_t = filtered_trial.T  # (S, 6)
            Y_t = Y_filt.T          # (S, 16)

            # 数值稳定：截断到有效 CCA 成分数
            actual_comp = min(n_comp, X_t.shape[1], Y_t.shape[1])
            if actual_comp < 1:
                all_scores[band_idx, cls_id] = 0.0
                continue

            cca = CCA(n_components=actual_comp)
            try:
                cca.fit(X_t, Y_t)
                X_c, Y_c = cca.transform(X_t, Y_t)
                r = np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1]
                all_scores[band_idx, cls_id] = abs(r)
            except Exception:
                all_scores[band_idx, cls_id] = 0.0

        # 子带内 z-score 归一化
        band_mean = all_scores[band_idx].mean()
        band_std = all_scores[band_idx].std()
        all_scores[band_idx] = (all_scores[band_idx] - band_mean) / (band_std + 1e-6)

    # FB 加权聚合
    scores = np.dot(ecca_refs["fb_weights"], all_scores)
    return scores


# ============================================================
# 辅助：集成预测（eCCA + eTRCA）
# ============================================================
def predict_ensemble(trial, ecca_refs, etrca_model):
    """
    FB-eCCA + FB-eTRCA 集成预测。

    score_final = 0.85 * normalized_ecca + 0.15 * normalized_etrca
    """
    score_ecca = predict_fb_ecca(trial, ecca_refs)
    score_etrca = predict_fb_etrca(trial, etrca_model)

    # min-max 归一化
    def norm_scores(s):
        s_min, s_max = s.min(), s.max()
        if s_max - s_min > 1e-10:
            return (s - s_min) / (s_max - s_min)
        return np.zeros_like(s)

    return FUSION_W_ECCA * norm_scores(score_ecca) + FUSION_W_ETRCA * norm_scores(score_etrca)


# ============================================================
# 辅助：计算准确率 & 混淆矩阵
# ============================================================
def evaluate(trials, ecca_refs, etrca_model=None):
    """
    对一批 trial 进行评估。

    参数
    ----
    trials : list of (eeg, stimID)
    ecca_refs : dict or None
    etrca_model : dict or None
        若为 None 则仅使用 eCCA；否则使用集成。

    返回
    ----
    acc : float
    cm : ndarray (8, 8)
    """
    correct = 0
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)

    for eeg, true_label in trials:
        if etrca_model is not None:
            scores = predict_ensemble(eeg, ecca_refs, etrca_model)
        else:
            scores = predict_fb_ecca(eeg, ecca_refs)
        pred = int(np.argmax(scores))
        if pred == true_label:
            correct += 1
        cm[true_label, pred] += 1

    acc = correct / len(trials) if trials else 0.0
    return acc, cm


# ============================================================
# 7. main —— 主流程
# ============================================================
def main():
    print("=" * 60)
    print("SSVEP 训练脚本 —— 小样本跨被试泛化方案")
    print("=" * 60)

    # ==========================================
    # 阶段 1：Offset 搜索（LOSO，仅用 FB-eCCA）
    # ==========================================
    print("\n[阶段 1] Offset 搜索（LOSO with FB-eCCA）\n")

    offsets_to_test = [0, 25, 50]
    best_offset = 0
    best_acc = 0.0
    offset_results = {}

    for offset in offsets_to_test:
        print(f"--- offset = {offset} ---")

        # 加载 D1、D2
        d1_trials = load_trials_by_taskid(
            f"{DATA_DIR}\\D1.csv", target_len=250, start_offset=offset
        )
        d2_trials = load_trials_by_taskid(
            f"{DATA_DIR}\\D2.csv", target_len=250, start_offset=offset
        )

        # 预处理
        d1_pp = [(preprocess_trial(eeg), lbl) for eeg, lbl in d1_trials]
        d2_pp = [(preprocess_trial(eeg), lbl) for eeg, lbl in d2_trials]

        # LOSO: D1 → D2
        d1_by_freq = {}
        for eeg, lbl in d1_pp:
            d1_by_freq.setdefault(lbl, []).append(eeg)
        ecca_d1 = build_fb_ecca_references(d1_by_freq, n_samples=250)
        acc_d1_to_d2, cm12 = evaluate(d2_pp, ecca_refs=ecca_d1, etrca_model=None)
        print(f"  D1→D2  acc = {acc_d1_to_d2:.4f}")

        # LOSO: D2 → D1
        d2_by_freq = {}
        for eeg, lbl in d2_pp:
            d2_by_freq.setdefault(lbl, []).append(eeg)
        ecca_d2 = build_fb_ecca_references(d2_by_freq, n_samples=250)
        acc_d2_to_d1, cm21 = evaluate(d1_pp, ecca_refs=ecca_d2, etrca_model=None)
        print(f"  D2→D1  acc = {acc_d2_to_d1:.4f}")

        avg_acc = (acc_d1_to_d2 + acc_d2_to_d1) / 2.0
        print(f"  平均    acc = {avg_acc:.4f}")
        offset_results[offset] = avg_acc

        if avg_acc > best_acc:
            best_acc = avg_acc
            best_offset = offset

    print(f"\n>>> 最佳 offset = {best_offset} (平均 acc = {best_acc:.4f})\n")

    # ==========================================
    # 阶段 2：全量训练（使用最佳 offset）
    # ==========================================
    print("[阶段 2] 全量训练（offset = {}）\n".format(best_offset))

    # 加载并预处理
    d1_trials = load_trials_by_taskid(
        f"{DATA_DIR}\\D1.csv", target_len=250, start_offset=best_offset
    )
    d2_trials = load_trials_by_taskid(
        f"{DATA_DIR}\\D2.csv", target_len=250, start_offset=best_offset
    )

    all_trials_raw = d1_trials + d2_trials
    all_trials_pp = [(preprocess_trial(eeg), lbl) for eeg, lbl in all_trials_raw]

    # CST 增强
    print("应用 CST 数据增强...")
    augmented_trials = []
    for eeg, lbl in all_trials_pp:
        aug_list = cyclic_shift_augment(eeg)
        for aug_eeg in aug_list:
            augmented_trials.append((aug_eeg, lbl))
    print(f"  增强前 trials: {len(all_trials_pp)} → 增强后: {len(augmented_trials)}")

    # 按频率分组
    train_by_freq = {}
    for eeg, lbl in augmented_trials:
        train_by_freq.setdefault(lbl, []).append(eeg)

    for cls_id in range(N_CLASSES):
        print(f"  class {cls_id} ({FREQS[cls_id]}Hz): {len(train_by_freq.get(cls_id, []))} trials")

    # 训练 FB-eTRCA
    print("\n训练 FB-eTRCA 模型...")
    etrca_model = train_fb_etrca(train_by_freq)

    # 构建 FB-eCCA 参考（使用增强前的数据，保持模板干净）
    clean_by_freq = {}
    for eeg, lbl in all_trials_pp:
        clean_by_freq.setdefault(lbl, []).append(eeg)
    ecca_refs = build_fb_ecca_references(clean_by_freq, n_samples=250)

    # ==========================================
    # 阶段 3：LOSO 验证（集成模型）
    # ==========================================
    print("\n[阶段 3] LOSO 验证（集成模型）\n")

    # D1→D2
    d1_pp = [(preprocess_trial(eeg), lbl) for eeg, lbl in d1_trials]
    d2_pp = [(preprocess_trial(eeg), lbl) for eeg, lbl in d2_trials]

    d1_by_freq = {}
    for eeg, lbl in d1_pp:
        d1_by_freq.setdefault(lbl, []).append(eeg)
    d1_ecca = build_fb_ecca_references(d1_by_freq, n_samples=250)

    # 在 D1 上训练 eTRCA（含增强）
    d1_aug = []
    for eeg, lbl in d1_pp:
        for aug_eeg in cyclic_shift_augment(eeg):
            d1_aug.append((aug_eeg, lbl))
    d1_aug_by_freq = {}
    for eeg, lbl in d1_aug:
        d1_aug_by_freq.setdefault(lbl, []).append(eeg)
    d1_etrca = train_fb_etrca(d1_aug_by_freq)

    acc_12_ecca, _ = evaluate(d2_pp, ecca_refs=d1_ecca, etrca_model=None)
    acc_12_ens, cm_12 = evaluate(d2_pp, ecca_refs=d1_ecca, etrca_model=d1_etrca)
    print(f"  D1→D2  eCCA only: {acc_12_ecca:.4f}  |  Ensemble: {acc_12_ens:.4f}")

    # D2→D1
    d2_by_freq = {}
    for eeg, lbl in d2_pp:
        d2_by_freq.setdefault(lbl, []).append(eeg)
    d2_ecca = build_fb_ecca_references(d2_by_freq, n_samples=250)

    d2_aug = []
    for eeg, lbl in d2_pp:
        for aug_eeg in cyclic_shift_augment(eeg):
            d2_aug.append((aug_eeg, lbl))
    d2_aug_by_freq = {}
    for eeg, lbl in d2_aug:
        d2_aug_by_freq.setdefault(lbl, []).append(eeg)
    d2_etrca = train_fb_etrca(d2_aug_by_freq)

    acc_21_ecca, _ = evaluate(d1_pp, ecca_refs=d2_ecca, etrca_model=None)
    acc_21_ens, cm_21 = evaluate(d1_pp, ecca_refs=d2_ecca, etrca_model=d2_etrca)
    print(f"  D2→D1  eCCA only: {acc_21_ecca:.4f}  |  Ensemble: {acc_21_ens:.4f}")

    print(f"\n  平均 eCCA : {(acc_12_ecca + acc_21_ecca) / 2:.4f}")
    print(f"  平均 Ens  : {(acc_12_ens + acc_21_ens) / 2:.4f}")

    # ==========================================
    # 阶段 4：保存模型
    # ==========================================
    print("\n[阶段 4] 保存模型\n")

    # --- 保存 FB-eTRCA 模型 ---
    etrca_path = f"{OUTPUT_DIR}\\model_1s_fb_etrca.pkl"
    with open(etrca_path, "wb") as f:
        pickle.dump(etrca_model, f)
    print(f"  已保存: {etrca_path}")

    # --- 保存 FB-eCCA 参数 ---
    ecca_params = {
        "n_harmonics": 5,
        "n_components": 2,
        "n_filterbanks": N_FILTERBANKS,
        "use_filterbank_weighting": True,
        "correction": False,
        "refs_by_freq": ecca_refs["refs_by_freq"],
        "filterbanks": ecca_refs["filterbanks"],
        "fb_weights": ecca_refs["fb_weights"],
    }
    ecca_path = f"{OUTPUT_DIR}\\fb_ecca_params.pkl"
    with open(ecca_path, "wb") as f:
        pickle.dump(ecca_params, f)
    print(f"  已保存: {ecca_path}")

    # ==========================================
    # 阶段 5：打印模型信息
    # ==========================================
    print("\n[阶段 5] 模型结构确认\n")
    print(f"  Best offset : {best_offset}")
    print(f"  Model keys  : {list(etrca_model.keys())}")
    print(f"  Bands count : {len(etrca_model['bands'])}")
    print(f"  FB weights  : {etrca_model['fb_weights'].round(4).tolist()}")

    for band_idx, band_info in enumerate(etrca_model["bands"]):
        W = band_info["W_ensemble"]
        print(f"\n  Band {band_idx} ({FILTERBANK_CUTS[band_idx]} Hz):")
        print(f"    W_ensemble shape : {W.shape}")
        for cls_id in sorted(band_info["templates"].keys()):
            t = band_info["templates"][cls_id]
            print(f"    template[{cls_id}] shape : {t.shape}")

    print(f"\n  Class count : {N_CLASSES}")
    print(f"  Freqs       : {FREQS}")
    print(f"\n{'=' * 60}")
    print("训练完成。")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
