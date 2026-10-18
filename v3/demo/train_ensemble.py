"""
SSVEP 比赛训练脚本 —— 频率感知解码优化版
=========================================
Task1（3s）：纯 FB-eCCA（不训练模型）
Task2（1s）：FB-eCCA + FB-eTRCA 频率感知集成

优化策略：
  ① 频率先验加权 — 抑制低频 dominance，防止高频系统性误判
  ② 混淆感知校正 — 对已知高频→低频混淆模式施加惩罚
  ③ 测试时校准  — 低置信度时降低 TRCA 权重，保持 eCCA 稳定性
  ④ 轻量 eTRCA — base weight 0.08，仅作为残差校正
  ⑤ 去被试 reference + LW shrinkage + eigenvalue clipping
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
FB_LAMBDA = 0.25                                           # LW shrinkage（增强正则）
CST_N_SHIFTS = 2                                           # 增强副本数
CST_SHIFT_RANGE = (15, 50)                                 # 移位范围（采样点）
FIXED_OFFSET = 50                                          # 固定 offset（~200ms latency）
EIGVAL_CLIP_MIN = 1e-3                                     # 协方差特征值裁剪下限
CALIBRATION_MARGIN_THRESHOLD = 0.10                          # 测试时校准的置信度阈值

# ---- 频率可靠性权重（基于 SSVEP 1s 窗口 SNR 分布） ----
# 低频信号在短窗口中 SNR 更高，高频 SNR 低，因此给予高频更低的先验权重
# 防止低频 dominance 导致高频系统性误判
_FREQ_RAW_WEIGHTS = {
    0: 1.2,   #  8 Hz — 最稳定
    1: 1.1,   #  9 Hz
    2: 1.1,   # 10 Hz
    3: 1.0,   # 11 Hz
    4: 1.0,   # 12 Hz
    5: 0.9,   # 13 Hz
    6: 0.7,   # 14 Hz — 开始退化
    7: 0.6,   # 15 Hz — 退化严重
}
_FREQ_WEIGHTS_SUM = sum(_FREQ_RAW_WEIGHTS.values())
FREQ_PRIOR_WEIGHTS = np.array(
    [_FREQ_RAW_WEIGHTS[i] / _FREQ_WEIGHTS_SUM for i in range(N_CLASSES)]
)  # 归一化到 sum=1

# ---- 混淆感知校正：已知高频易混淆到低频的配对 ----
# 来源：LOSO 观察；格式 (pred→true 混淆, penalty)
HIGH_CONFUSION_PAIRS = {
    # 14 Hz 常被误判为 10 Hz（二次谐波混淆）
    (6, 2): 0.15,
    # 15 Hz 常被误判为 8 Hz（接近二次谐波）
    (7, 0): 0.18,
    # 13 Hz 常被误判为低频
    (5, 1): 0.10,
    (6, 1): 0.10,
    (7, 1): 0.12,
}

# 数据路径
DATA_DIR = r"C:\Users\Administrator\Desktop\v3\ExampleData"
OUTPUT_DIR = r"C:\Users\Administrator\Desktop\v3\demo"


# ============================================================
# 辅助：构建数据驱动的频率先验权重
# ============================================================
def build_frequency_prior_weights(per_class_acc, base_weights=FREQ_PRIOR_WEIGHTS):
    """
    结合 LOSO per-class accuracy 与先验权重，生成频率可靠性权重。

    将 LOSO 观察到的各类准确率作为自适应因子，与先验权重相乘后归一化。
    这确保低频类（高 acc）不会过度主导，同时高频类（低 acc）获得相对提升。

    参数
    ----
    per_class_acc : ndarray, shape (8,)
        LOSO 中每个频率类的正确率。
    base_weights : ndarray, shape (8,)
        先验基础权重。

    返回
    ----
    freq_weights : ndarray, shape (8,)
        归一化后的频率权重。
    """
    # 将准确率映射到 [0.5, 1.5] 范围作为自适应因子
    # 高准确率 → 因子接近 1.0（不额外增强）
    # 低准确率 → 因子接近 1.5（需要更多"帮助"，但通过权重降低实现）
    acc_clipped = np.clip(per_class_acc, 0.1, 1.0)
    # 低准确率需要更大的惩罚 → 因子 < 1
    adaptive_factor = np.clip(acc_clipped / acc_clipped.max(), 0.5, 1.0)

    combined = base_weights * adaptive_factor
    combined = combined / combined.sum()
    return combined


# ============================================================
# 辅助：从混淆矩阵构建校正偏置
# ============================================================
def build_confusion_correction(cm):
    """
    从 LOSO 混淆矩阵中提取系统性混淆模式，构建校正映射。

    对于混淆矩阵中高频→低频的显著误判，返回一个校正 dict：
      (pred_class, true_class) → penalty
    用于在推理时对可疑预测施加惩罚。

    参数
    ----
    cm : ndarray, shape (8, 8)
        混淆矩阵，cm[true, pred]。

    返回
    ----
    correction_map : dict
        {(pred_class, true_class): penalty}
    """
    correction_map = {}
    n = cm.sum()
    if n == 0:
        return correction_map

    for true_cls in range(N_CLASSES):
        row_total = cm[true_cls].sum()
        if row_total == 0:
            continue
        for pred_cls in range(N_CLASSES):
            if pred_cls == true_cls:
                continue
            # 误判比例
            err_rate = cm[true_cls, pred_cls] / row_total
            # 仅对显著混淆（>10%）且高频→低频添加校正
            if err_rate > 0.10 and FREQS[pred_cls] < FREQS[true_cls]:
                penalty = min(err_rate * 0.8, 0.25)  # 限制最大惩罚
                correction_map[(true_cls, pred_cls)] = penalty

    return correction_map


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

            # ----- 特征值裁剪（防止病态矩阵） -----
            eigvals_Q, eigvecs_Q = np.linalg.eigh(Q_f)
            eigvals_Q = np.clip(eigvals_Q, EIGVAL_CLIP_MIN, None)
            Q_f = eigvecs_Q @ np.diag(eigvals_Q) @ eigvecs_Q.T

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
            # sign(r) * r^2，裁剪到 [-1, 1]
            all_scores[band_idx, cls_id] = np.clip(np.sign(r) * (r ** 2), -1.0, 1.0)

        # 子带内 z-score 归一化
        band_mean = all_scores[band_idx].mean()
        band_std = all_scores[band_idx].std()
        all_scores[band_idx] = (all_scores[band_idx] - band_mean) / (band_std + 1e-6)

    # FB 加权聚合
    scores = np.dot(model["fb_weights"], all_scores)  # (8,)
    return scores


# ============================================================
# 6. build_fb_ecca_references —— 构建去被试 FB-eCCA 参考
# ============================================================
def build_fb_ecca_references(train_data_by_freq, d1_data_by_freq=None,
                             d2_data_by_freq=None, n_samples=250, srate=SRATE):
    """
    构建去被试偏差的 FB-eCCA 增强参考信号。

    每个频率的参考由三部分拼接：
      1. 理论 sin/cos 模板（5 次谐波，10 维）
      2. 全局模板 = 0.7*cross_subj_mean + 0.3*subj_norm_mean（6 维）
      3. 被试归一化模板 × 0.3（6 维）
    总维度 = 22 维。

    对 5 个子带分别滤波 + 独立归一化。

    参数
    ----
    train_data_by_freq : dict
        {class_id: [trial, ...]}，所有训练 trial（D1+D2 合并）。
    d1_data_by_freq : dict or None
        D1 独立的 trial 分组（用于被试归一化）。若为 None 则等价于 train_data_by_freq。
    d2_data_by_freq : dict or None
        D2 独立的 trial 分组。

    返回
    ----
    ecca_refs : dict
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

    # ----- 缺省处理 -----
    if d1_data_by_freq is None:
        d1_data_by_freq = train_data_by_freq
    if d2_data_by_freq is None:
        d2_data_by_freq = train_data_by_freq

    # ----- 构建每个频率的去被试参考 -----
    refs_by_freq = {}
    for cls_id in sorted(train_data_by_freq.keys()):
        freq = FREQS[cls_id]

        # ① 理论模板：5 次谐波 sin + cos → (10, n_samples)
        harmonics = []
        for h in range(1, 6):
            harmonics.append(np.sin(2 * np.pi * freq * h * t))
            harmonics.append(np.cos(2 * np.pi * freq * h * t))
        sin_cos = np.vstack(harmonics)  # (10, n_samples)

        # ② 跨被试均值（D1+D2 所有 trial 平均）
        trials_all = train_data_by_freq[cls_id]
        cross_subj_mean = np.mean(trials_all, axis=0)  # (6, n_samples)

        # ③ 被试归一化模板
        def _subj_norm_template(data_by_freq, cls_id):
            """对单个被试的该类模板做 z-score 归一化（全局均值/方差）"""
            if cls_id in data_by_freq and len(data_by_freq[cls_id]) > 0:
                avg = np.mean(data_by_freq[cls_id], axis=0)  # (6, n_samples)
                m = avg.mean()
                s = avg.std()
                return (avg - m) / (s + 1e-6)
            return np.zeros((N_CHANS, n_samples))

        d1_norm = _subj_norm_template(d1_data_by_freq, cls_id)
        d2_norm = _subj_norm_template(d2_data_by_freq, cls_id)
        subj_norm_mean = (d1_norm + d2_norm) / 2.0  # (6, n_samples)

        # ④ 全局模板（去被试混合）
        alpha, beta = 0.7, 0.3
        global_template = alpha * cross_subj_mean + beta * subj_norm_mean  # (6, n_samples)

        # ⑤ 拼接参考（22 维）
        Y_ref = np.vstack([
            sin_cos,                       # 10 维：理论谐波
            global_template,               #  6 维：去被试全局模板
            0.3 * subj_norm_mean,          #  6 维：幅度抑制的被试归一化模板
        ])  # (22, n_samples)
        refs_by_freq[cls_id] = Y_ref

    # ----- 对每个子带做独立 reference normalization -----
    refs_normalized = {}
    for cls_id in sorted(refs_by_freq.keys()):
        ref_per_band = []
        for b_band, a_band in filterbanks:
            ref_filt = signal.filtfilt(b_band, a_band, refs_by_freq[cls_id], axis=1)
            # 子带内 z-score 归一化
            ref_mean = ref_filt.mean(axis=1, keepdims=True)
            ref_std = ref_filt.std(axis=1, keepdims=True)
            ref_filt = (ref_filt - ref_mean) / (ref_std + 1e-6)
            ref_per_band.append(ref_filt)
        refs_normalized[cls_id] = ref_per_band

    ecca_refs = {
        "refs_by_freq": refs_by_freq,           # 原始参考 (22, n_samples)
        "refs_by_freq_norm": refs_normalized,   # 子带归一化参考 {cls_id: [band0, ...]}
        "filterbanks": filterbanks,
        "fb_weights": fb_weights,
        "n_harmonics": 5,
        "n_components": 2,
    }
    return ecca_refs


# ============================================================
# 辅助：FB-eCCA 预测
# ============================================================
def predict_fb_ecca(trial, ecca_refs, freq_weights=None):
    """
    使用 FB-eCCA 对单个 trial 进行预测（频率感知版）。

    频率先验以凸组合方式融合（非乘法），防止过度压制低频：
      final = (1 - alpha) * norm_score + alpha * freq_weights
    其中 alpha = 0.15，确保模型分数始终主导决策。

    参数
    ----
    trial : ndarray, shape (6, samples)
    ecca_refs : dict
    freq_weights : ndarray (8,) or None
        频率可靠性权重。若为 None 则跳过。

    返回
    ----
    scores : ndarray, shape (8,)
        [0,1] 归一化得分。
    raw_scores : ndarray, shape (8,)
        频率先验加权前的原始聚合得分（用于置信度计算）。
    """
    n_comp = ecca_refs["n_components"]
    all_scores = np.zeros((N_FILTERBANKS, N_CLASSES))
    use_precomputed = "refs_by_freq_norm" in ecca_refs

    for band_idx, (b_band, a_band) in enumerate(ecca_refs["filterbanks"]):
        # 滤波试次
        filtered_trial = signal.filtfilt(b_band, a_band, trial, axis=1)  # (6, S)

        for cls_id in range(N_CLASSES):
            if use_precomputed:
                # 直接使用预计算并归一化的参考
                Y_filt = ecca_refs["refs_by_freq_norm"][cls_id][band_idx]  # (22, S)
                # 试次也需要对应归一化
                X_for_cca = filtered_trial  # (6, S)
            else:
                # 回退：实时滤波参考
                Y_ref = ecca_refs["refs_by_freq"][cls_id]  # (K, S)
                Y_filt = signal.filtfilt(b_band, a_band, Y_ref, axis=1)
                X_for_cca = filtered_trial

            # CCA：X^T (S, 6)，Y^T (S, K)
            X_t = X_for_cca.T
            Y_t = Y_filt.T

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

    scores = np.dot(ecca_refs["fb_weights"], all_scores)  # (8,)

    # ---- 频率先验：凸组合融合（温和调整） ----
    if freq_weights is not None:
        FREQ_PRIOR_ALPHA = 0.15  # 频率先验占比（保守）
        # min-max 归一化模型得分
        s_norm = _minmax_norm(scores)
        # 凸组合：模型得分主导 (85%)，频率先验微调 (15%)
        scores = (1.0 - FREQ_PRIOR_ALPHA) * s_norm + FREQ_PRIOR_ALPHA * freq_weights

    return scores, scores.copy()


# ============================================================
# 辅助：自适应集成预测（eCCA + eTRCA）
# ============================================================
def predict_ensemble(trial, ecca_refs, etrca_model, freq_weights=None,
                     correction_map=None):
    """
    FB-eCCA + FB-eTRCA 自适应集成预测（频率感知 + 测试时校准版）。

    流程：
      1. eCCA 得分 × 频率先验 → softmax
      2. eTRCA 得分归一化
      3. 测试时校准：若 eCCA score margin < threshold → 降低 eTRCA 权重
      4. 混淆感知校正：对已知高频混淆配对施加惩罚
      5. 最终融合：eCCA + 0.1 * eTRCA（基础权重）

    参数
    ----
    trial : ndarray, shape (6, samples)
    ecca_refs : dict
    etrca_model : dict
    freq_weights : ndarray (8,) or None
    correction_map : dict or None
        {(true_class, confused_pred): penalty}

    返回
    ----
    scores : ndarray (8,)
    w_ecca : float
    w_etrca : float
    margin : float         (校准用 score margin)
    calibrated : bool       (是否触发校准)
    """
    # ① eCCA：频率先验加权
    score_ecca_soft, score_ecca_raw = predict_fb_ecca(trial, ecca_refs, freq_weights)

    # ② eTRCA：独立归一化
    score_etrca = predict_fb_etrca(trial, etrca_model)
    s_etrca_norm = _minmax_norm(score_etrca)

    # ③ 测试时校准：基于 eCCA score margin
    top2_idx = np.argpartition(score_ecca_soft, -2)[-2:]
    margin = score_ecca_soft[top2_idx[1]] - score_ecca_soft[top2_idx[0]]

    if margin < CALIBRATION_MARGIN_THRESHOLD:
        # 低置信度 → 降低 eTRCA 影响，更依赖 eCCA
        w_etrca = 0.05
        calibrated = True
    else:
        # 正常置信度
        w_etrca = 0.10  # 固定轻量权重，不做动态调整
        calibrated = False

    w_ecca = 1.0 - w_etrca

    # ④ 基础融合
    scores = w_ecca * score_ecca_soft + w_etrca * s_etrca_norm

    # ⑤ 混淆感知校正：仅当混淆候选在当前得分中也显著时生效
    if correction_map is not None:
        pred_cls = int(np.argmax(scores))
        for (true_cls, confused_pred), penalty in correction_map.items():
            if pred_cls == confused_pred and scores[true_cls] > scores[pred_cls] * 0.4:
                # 混淆目标存在合理备选时才惩罚
                scores[confused_pred] -= penalty * 0.5
                scores = np.clip(scores, 0.0, None)

    return scores, w_ecca, w_etrca, margin, calibrated


def _minmax_norm(s):
    """min-max 归一化到 [0, 1]"""
    s_min, s_max = s.min(), s.max()
    if s_max - s_min > 1e-10:
        return (s - s_min) / (s_max - s_min)
    return np.zeros_like(s)


# ============================================================
# 辅助：评估函数（含 per-class accuracy & confusion matrix）
# ============================================================
def evaluate(trials, ecca_refs, etrca_model=None, verbose=True,
             label_ecca="eCCA", label_ens="Ensemble",
             freq_weights=None, correction_map=None,
             report_high_freq=False):
    """
    对一批 trial 进行评估（频率感知版）。

    参数
    ----
    trials : list of (eeg, stimID)
    ecca_refs : dict
    etrca_model : dict or None
    verbose : bool
    label_ecca, label_ens : str
    freq_weights : ndarray (8,) or None
    correction_map : dict or None
    report_high_freq : bool
        是否输出高频（13-15Hz）改进报告。

    返回
    ----
    acc_final : float          最终准确率
    cm : ndarray (8, 8)       混淆矩阵
    per_class_acc : ndarray (8,)
    acc_ecca : float or None
    acc_ens : float or None    集成准确率
    high_freq_report : dict    {freq: {'ecca': acc, 'ens': acc}} or None
    """
    correct_ecca = 0
    correct_ens = 0
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    cm_ecca = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    weights_log = []
    calib_count = 0

    for eeg, true_label in trials:
        # eCCA 预测
        score_ecca_soft, _ = predict_fb_ecca(eeg, ecca_refs, freq_weights)
        pred_ecca = int(np.argmax(score_ecca_soft))
        if pred_ecca == true_label:
            correct_ecca += 1
        cm_ecca[true_label, pred_ecca] += 1

        # 集成预测
        if etrca_model is not None:
            score_ens, w_ecca, w_etrca, margin, calibrated = predict_ensemble(
                eeg, ecca_refs, etrca_model, freq_weights, correction_map
            )
            weights_log.append((w_ecca, w_etrca, margin))
            if calibrated:
                calib_count += 1
            pred_final = int(np.argmax(score_ens))
        else:
            pred_final = pred_ecca

        if pred_final == true_label:
            correct_ens += 1
        cm[true_label, pred_final] += 1

    n = len(trials)
    acc_ecca = correct_ecca / n if n else 0.0
    acc_ens = correct_ens / n if n else 0.0

    # Per-class accuracy
    per_class_acc = np.zeros(N_CLASSES)
    per_class_acc_ecca = np.zeros(N_CLASSES)
    for cls_id in range(N_CLASSES):
        row_sum = cm[cls_id].sum()
        if row_sum > 0:
            per_class_acc[cls_id] = cm[cls_id, cls_id] / row_sum
        row_sum_ecca = cm_ecca[cls_id].sum()
        if row_sum_ecca > 0:
            per_class_acc_ecca[cls_id] = cm_ecca[cls_id, cls_id] / row_sum_ecca

    # 高频改进报告
    high_freq_report = None
    if report_high_freq:
        high_freq_ids = [5, 6, 7]  # 13, 14, 15 Hz
        high_freq_report = {}
        for cls_id in high_freq_ids:
            high_freq_report[cls_id] = {
                "freq": FREQS[cls_id],
                "ecca_acc": per_class_acc_ecca[cls_id],
                "ens_acc": per_class_acc[cls_id],
                "improvement": per_class_acc[cls_id] - per_class_acc_ecca[cls_id],
            }

    if verbose:
        if etrca_model is not None:
            avg_w = np.mean(weights_log, axis=0) if weights_log else (1.0, 0.0, 0.0)
            avg_margin = avg_w[2] if len(avg_w) > 2 else 0.0
            print(f"  {label_ecca} acc : {acc_ecca:.4f}")
            print(f"  {label_ens} acc : {acc_ens:.4f}  (gap: {acc_ens - acc_ecca:+.4f})")
            print(f"  Avg w         : eCCA={avg_w[0]:.3f} / eTRCA={avg_w[1]:.3f}")
            print(f"  Avg margin    : {avg_margin:.3f}")
            print(f"  Calibrated    : {calib_count}/{n} trials ({100*calib_count/n:.1f}%)")
            print(f"  Per-class Ens : {' '.join(f'{v:.2f}' for v in per_class_acc)}")
            print(f"  Per-class eCCA: {' '.join(f'{v:.2f}' for v in per_class_acc_ecca)}")

            if report_high_freq and high_freq_report:
                print(f"\n  >>> 高频改进报告 (13-15Hz) <<<")
                for cls_id in sorted(high_freq_report.keys()):
                    r = high_freq_report[cls_id]
                    print(f"    {r['freq']}Hz: eCCA={r['ecca_acc']:.2f} → Ens={r['ens_acc']:.2f} "
                          f"({'↑' if r['improvement'] >= 0 else '↓'}{abs(r['improvement']):+.2f})")
                hf_ecca_avg = np.mean([high_freq_report[c]["ecca_acc"] for c in high_freq_ids])
                hf_ens_avg = np.mean([high_freq_report[c]["ens_acc"] for c in high_freq_ids])
                print(f"    高频平均: eCCA={hf_ecca_avg:.2f} → Ens={hf_ens_avg:.2f} "
                      f"({'↑' if hf_ens_avg >= hf_ecca_avg else '↓'}{abs(hf_ens_avg - hf_ecca_avg):+.2f})")

            print(f"  Confusion matrix (Ensemble):")
            print_cm(cm)
        else:
            print(f"  {label_ecca} acc : {acc_ecca:.4f}")
            print(f"  Per-class   : {' '.join(f'{v:.2f}' for v in per_class_acc)}")

    return acc_ens, cm, per_class_acc, acc_ecca, acc_ens, high_freq_report


def print_cm(cm):
    """打印紧凑混淆矩阵"""
    header = "     " + " ".join(f"p{f}" for f in range(N_CLASSES))
    print(header)
    for i in range(N_CLASSES):
        row = " ".join(f"{cm[i, j]:3d}" for j in range(N_CLASSES))
        print(f"  t{i}: {row}")


# ============================================================
# 7. main —— 主流程（稳定化优化版）
# ============================================================
def main():
    print("=" * 60)
    print("SSVEP 训练脚本 —— 频率感知解码优化版")
    print("  ① 频率先验加权 ② 混淆感知校正 ③ 测试时校准 ④ 轻量 eTRCA")
    print("=" * 60)

    # ==========================================
    # 阶段 1：加载数据（固定 offset=50）
    # ==========================================
    print(f"\n[阶段 1] 加载数据（offset = {FIXED_OFFSET}）\n")

    d1_trials = load_trials_by_taskid(
        f"{DATA_DIR}\\D1.csv", target_len=250, start_offset=FIXED_OFFSET
    )
    d2_trials = load_trials_by_taskid(
        f"{DATA_DIR}\\D2.csv", target_len=250, start_offset=FIXED_OFFSET
    )
    print(f"  D1 trials: {len(d1_trials)}")
    print(f"  D2 trials: {len(d2_trials)}")

    # 预处理
    d1_pp = [(preprocess_trial(eeg), lbl) for eeg, lbl in d1_trials]
    d2_pp = [(preprocess_trial(eeg), lbl) for eeg, lbl in d2_trials]

    # 按频率分组（保留被试信息）
    d1_by_freq = {}
    for eeg, lbl in d1_pp:
        d1_by_freq.setdefault(lbl, []).append(eeg)
    d2_by_freq = {}
    for eeg, lbl in d2_pp:
        d2_by_freq.setdefault(lbl, []).append(eeg)

    # ==========================================
    # 阶段 2：LOSO 验证（仅 eCCA，评估基线 & 构建频率校正）
    # ==========================================
    print("\n[阶段 2] LOSO 基线评估 & 频率校正构建\n")

    # D1 → D2
    ecca_d1 = build_fb_ecca_references(d1_by_freq, d1_data_by_freq=d1_by_freq,
                                       d2_data_by_freq=None, n_samples=250)
    _, cm_d1_to_d2, pc_d12, acc_ecca_12, _, _ = evaluate(
        d2_pp, ecca_refs=ecca_d1, etrca_model=None,
        label_ecca="D1→D2 eCCA", label_ens="", verbose=False
    )

    # D2 → D1
    ecca_d2 = build_fb_ecca_references(d2_by_freq, d1_data_by_freq=None,
                                       d2_data_by_freq=d2_by_freq, n_samples=250)
    _, cm_d2_to_d1, pc_d21, acc_ecca_21, _, _ = evaluate(
        d1_pp, ecca_refs=ecca_d2, etrca_model=None,
        label_ecca="D2→D1 eCCA", label_ens="", verbose=False
    )

    print(f"  D1→D2 eCCA acc : {acc_ecca_12:.4f}")
    print(f"  D2→D1 eCCA acc : {acc_ecca_21:.4f}")
    print(f"  平均 eCCA      : {(acc_ecca_12 + acc_ecca_21) / 2:.4f}")

    # ---- 构建频率先验权重（基于 LOSO per-class accuracy） ----
    avg_per_class = (pc_d12 + pc_d21) / 2.0
    freq_weights = build_frequency_prior_weights(avg_per_class)
    print(f"\n  频率先验权重   : {' '.join(f'{w:.3f}' for w in freq_weights)}")
    print(f"  各类 eCCA 准确率: {' '.join(f'{v:.2f}' for v in avg_per_class)}")

    # ---- 构建混淆校正（基于合并的 LOSO 混淆矩阵） ----
    cm_combined = cm_d1_to_d2 + cm_d2_to_d1
    correction_map = build_confusion_correction(cm_combined)
    if correction_map:
        print(f"  混淆校正对     : {len(correction_map)} 对")
        for (tc, cp), pen in correction_map.items():
            print(f"    true={tc}({FREQS[tc]}Hz) <-> pred={cp}({FREQS[cp]}Hz) penalty={pen:.3f}")
    else:
        print(f"  混淆校正对     : 0（无需校正）")

    # ==========================================
    # 阶段 3：全量训练（D1 + D2）
    # ==========================================
    print("\n[阶段 3] 全量训练（D1 + D2，offset = {}）\n".format(FIXED_OFFSET))

    all_trials_pp = d1_pp + d2_pp

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
        n_trials = len(train_by_freq.get(cls_id, []))
        print(f"  class {cls_id} ({FREQS[cls_id]}Hz): {n_trials} trials")

    # 训练 FB-eTRCA
    print("\n训练 FB-eTRCA 模型（λ={}, clip={:.0e}）...".format(FB_LAMBDA, EIGVAL_CLIP_MIN))
    etrca_model = train_fb_etrca(train_by_freq)

    # 构建 FB-eCCA 参考（使用增强前数据，含被试信息用于去偏）
    clean_by_freq = {}
    for eeg, lbl in all_trials_pp:
        clean_by_freq.setdefault(lbl, []).append(eeg)
    ecca_refs = build_fb_ecca_references(
        clean_by_freq,
        d1_data_by_freq=d1_by_freq,
        d2_data_by_freq=d2_by_freq,
        n_samples=250,
    )

    # ==========================================
    # 阶段 4：LOSO 验证（频率感知集成模型）
    # ==========================================
    print("\n[阶段 4] LOSO 验证（频率感知集成模型）\n")

    # ---- D1 → D2 ----
    d1_aug = []
    for eeg, lbl in d1_pp:
        for aug_eeg in cyclic_shift_augment(eeg):
            d1_aug.append((aug_eeg, lbl))
    d1_aug_by_freq = {}
    for eeg, lbl in d1_aug:
        d1_aug_by_freq.setdefault(lbl, []).append(eeg)
    d1_etrca = train_fb_etrca(d1_aug_by_freq)

    print("--- D1 → D2 ---")
    _, _, _, _, _, hf_report_12 = evaluate(
        d2_pp, ecca_refs=ecca_d1, etrca_model=d1_etrca,
        label_ecca="eCCA", label_ens="Ensemble",
        freq_weights=freq_weights, correction_map=correction_map,
        report_high_freq=True,
    )

    # ---- D2 → D1 ----
    d2_aug = []
    for eeg, lbl in d2_pp:
        for aug_eeg in cyclic_shift_augment(eeg):
            d2_aug.append((aug_eeg, lbl))
    d2_aug_by_freq = {}
    for eeg, lbl in d2_aug:
        d2_aug_by_freq.setdefault(lbl, []).append(eeg)
    d2_etrca = train_fb_etrca(d2_aug_by_freq)

    print("\n--- D2 → D1 ---")
    _, _, _, _, _, hf_report_21 = evaluate(
        d1_pp, ecca_refs=ecca_d2, etrca_model=d2_etrca,
        label_ecca="eCCA", label_ens="Ensemble",
        freq_weights=freq_weights, correction_map=correction_map,
        report_high_freq=True,
    )

    # ==========================================
    # 阶段 5：保存模型（含频率感知参数）
    # ==========================================
    print("\n[阶段 5] 保存模型（含频率感知参数）\n")

    etrca_path = f"{OUTPUT_DIR}\\model_1s_fb_etrca.pkl"
    etrca_model["freq_weights"] = freq_weights
    etrca_model["correction_map"] = correction_map
    etrca_model["calibration_threshold"] = CALIBRATION_MARGIN_THRESHOLD
    with open(etrca_path, "wb") as f:
        pickle.dump(etrca_model, f)
    print(f"  已保存: {etrca_path}")
    print(f"    + freq_weights: {' '.join(f'{w:.3f}' for w in freq_weights)}")
    print(f"    + correction_map: {len(correction_map)} pairs")

    # FB-eCCA 参数（保存原始参考 + 子带归一化参考 + 频率先验）
    ecca_params = {
        "n_harmonics": 5,
        "n_components": 2,
        "n_filterbanks": N_FILTERBANKS,
        "use_filterbank_weighting": True,
        "correction": False,
        "refs_by_freq": ecca_refs["refs_by_freq"],
        "refs_by_freq_norm": ecca_refs.get("refs_by_freq_norm", {}),
        "filterbanks": ecca_refs["filterbanks"],
        "fb_weights": ecca_refs["fb_weights"],
        "freq_weights": freq_weights,
        "correction_map": correction_map,
        "calibration_threshold": CALIBRATION_MARGIN_THRESHOLD,
    }
    ecca_path = f"{OUTPUT_DIR}\\fb_ecca_params.pkl"
    with open(ecca_path, "wb") as f:
        pickle.dump(ecca_params, f)
    print(f"  已保存: {ecca_path}")

    # ==========================================
    # 阶段 6：模型结构确认 & 频率感知摘要
    # ==========================================
    print("\n[阶段 6] 模型结构确认 & 频率感知摘要\n")
    print(f"  Offset         : {FIXED_OFFSET}")
    print(f"  Shrinkage λ    : {FB_LAMBDA}")
    print(f"  Eigval clip    : {EIGVAL_CLIP_MIN:.0e}")
    print(f"  Calib threshold: {CALIBRATION_MARGIN_THRESHOLD}")
    print(f"  Model keys     : {list(etrca_model.keys())}")
    print(f"  Bands count    : {len(etrca_model['bands'])}")
    print(f"  FB weights     : {etrca_model['fb_weights'].round(4).tolist()}")
    print(f"  Freq weights   : {' '.join(f'{w:.3f}' for w in freq_weights)}")
    print(f"  Correction map : {len(correction_map)} pairs")
    print(f"  eCCA ref dim   : {ecca_refs['refs_by_freq'][0].shape[0]} (22 = 10 sin/cos + 6 global + 6 subj-norm)")

    for band_idx, band_info in enumerate(etrca_model["bands"]):
        W = band_info["W_ensemble"]
        print(f"\n  Band {band_idx} ({FILTERBANK_CUTS[band_idx]} Hz):")
        print(f"    W_ensemble shape : {W.shape}")
        t0 = band_info["templates"][0]
        print(f"    template shape   : {t0.shape} (×{len(band_info['templates'])} classes)")

    print(f"\n  Class count : {N_CLASSES}")
    print(f"  Freqs       : {FREQS}")

    # 高频改进汇总
    print(f"\n  >>> 频率感知解码优化摘要 <<<")
    print(f"  策略 1: 频率先验加权 (freq_weights)  — 抑制低频 dominance")
    print(f"  策略 2: 混淆感知校正 (correction)    — 惩罚高频→低频误判")
    print(f"  策略 3: 测试时校准 (margin<{CALIBRATION_MARGIN_THRESHOLD}) — 低置信度降低 TRCA")
    print(f"  策略 4: TRCA base weight={0.08:.2f}  — 轻量残差校正")
    if correction_map:
        print(f"  已知高频混淆模式:")
        for (tc, cp), pen in correction_map.items():
            print(f"    {FREQS[tc]}Hz → {FREQS[cp]}Hz  penalty={pen:.3f}")

    print(f"\n{'=' * 60}")
    print("训练完成 —— 频率感知解码优化版。")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
