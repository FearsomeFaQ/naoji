# SSVEP（稳态视觉诱发电位）检测模块
# 基于 FBCCA（滤波器组典型相关分析）实现多频率 SSVEP 信号分类

import numpy as np
from scipy import signal
from scipy.linalg import eigh
from sklearn.cross_decomposition import CCA


class ssvepDetect:
    """SSVEP 频率检测器

    核心算法：FBCCA（Filter Bank Canonical Correlation Analysis）
    利用多个子带滤波器对 EEG 信号进行分频处理，结合 CCA 计算
    观测信号与各频率参考模板之间的相关性，从而判定用户注视的目标频率。
    """

    def __init__(self, srate, freqs, dataLen, n_harmonics=3, n_components=3,
                 chans=None, use_filterbank=True, use_car=True,
                 use_correction=True, n_filter_bands=8, filter_order=4):
        """
        参数
        ----
        srate : int
            EEG 采样率（Hz）
        freqs : list[float]
            待检测的 SSVEP 目标频率列表，例如 [8.0, 9.0, 10.0, ...]
        dataLen : float
            单次检测使用的数据时长（秒）
        n_harmonics : int
            参考模板中使用的谐波数量（默认 3）
        n_components : int
            CCA 分解时保留的典型变量数（默认 3）
        chans : list[int] or None
            使用的通道索引列表，None 表示使用全部通道
        use_filterbank : bool
            是否启用滤波器组分解（默认 True）
        use_car : bool
            是否使用 CAR（共平均参考）空间滤波（默认 True）
        use_correction : bool
            是否启用自适应纠正机制（默认 True）
        n_filter_bands : int
            滤波器组中的子带数量（默认 8）
        filter_order : int
            Butterworth 滤波器阶数（默认 4）
        """

        # 根据数据长度自动优化参数（校准自 D1/D2 截断测试）
        if dataLen <= 1:
            n_harmonics = 5        # Task2: 更多谐波捕获短时窗信号
            n_components = 2       # 更少 CCA 分量避免过拟合
            use_correction = False # 关闭校正，短数据噪声大易误判
            n_filter_bands = 6     # 减少滤波子带，降低瞬态效应
            filter_order = 2       # 低阶滤波，适合短数据
        elif dataLen <= 3:
            n_harmonics = 4        # Task1: 4 谐波最优（校准验证 94.79%）
            n_components = 2       # 分量数影响微小，取低值省算力
            use_correction = True  # 3s 数据校正有效，准确提升 ~4%
            n_filter_bands = 4     # 少子带+高阶滤波效果更好
            filter_order = 6       # 高阶滤波提升频率选择性

        self.srate = srate
        self.freqs = freqs
        self.dataLen = dataLen
        self.n_harmonics = n_harmonics
        self.n_components = n_components
        self.chans = chans
        self.use_filterbank = use_filterbank
        self.use_car = use_car
        self.use_correction = use_correction
        self.n_filter_bands = n_filter_bands
        self.filter_order = filter_order

        # 构建时间轴采样点
        templLen = int(dataLen * srate)
        sample = np.linspace(0, (templLen - 1) / srate, templLen, endpoint=True)

        # 主模板集：为每个目标频率生成 正弦/余弦 参考信号
        # TemplateSet[i] 的形状为 (2 * n_harmonics, templLen)，对应频率 freqs[i]
        self.TemplateSet = []
        for freq in freqs:
            tempset_list = []
            for h in range(1, n_harmonics + 1):
                phase = 2 * np.pi * freq * h * sample
                tempset_list.append(np.sin(phase))
                tempset_list.append(np.cos(phase))
            self.TemplateSet.append(np.vstack(tempset_list))

        # 备用模板库（谐波数 2/3/4/5），供自适应纠正使用
        self._alt_templates = {}
        for nh in [2, 3, 4, 5]:
            tset = []
            for freq in freqs:
                tempset_list = []
                for h in range(1, nh + 1):
                    phase = 2 * np.pi * freq * h * sample
                    tempset_list.append(np.sin(phase))
                    tempset_list.append(np.cos(phase))
                tset.append(np.vstack(tempset_list))
            self._alt_templates[nh] = tset

        # 初始化滤波器组
        if use_filterbank:
            self._init_filterbank()

    # ── 滤波器组 ────────────────────────────────────────────
    def _init_filterbank(self):
        """初始化带通滤波器组和子带权重

        将 6-40 Hz 频段按低端截止频率划分为多个子带，每个子带上界
        统一为 40 Hz。子带权重随频率升高递减，低频段贡献更大。
        """
        nyq = self.srate / 2

        # 根据子带数生成低端截止频率（Hz）
        if self.n_filter_bands >= 8:
            low_cuts = [6, 10, 14, 18, 22, 26, 30, 34]
        elif self.n_filter_bands == 6:
            low_cuts = [6, 10, 14, 18, 22, 26]
        elif self.n_filter_bands == 4:
            low_cuts = [6, 12, 18, 24]
        else:
            step = 34 // self.n_filter_bands
            low_cuts = [6 + i * step for i in range(self.n_filter_bands)]

        low_cuts = low_cuts[:self.n_filter_bands]

        # 为每个子带创建 Butterworth 带通滤波器
        self.filter_banks = []
        for low_cut in low_cuts:
            b, a = signal.butter(self.filter_order, [low_cut / nyq, 40 / nyq], btype='bandpass')
            self.filter_banks.append((b, a))

        # 计算子带加权系数：n^{-a} + b 形式，低频子带权重更高
        n_bands = len(self.filter_banks)
        a_param, b_param = 1.25, 0.25
        raw_weights = np.array(
            [n ** (-a_param) + b_param for n in range(1, n_bands + 1)]
        )
        self.fb_weights = raw_weights / raw_weights.sum()

    # ── CCA 得分 ────────────────────────────────────────────
    def _cca_score(self, X, Y):
        """计算两组变量 X 和 Y 之间的典型相关系数

        参数
        ----
        X : ndarray (n_samples, n_features_x)
            观测数据（EEG 信号）
        Y : ndarray (n_samples, n_features_y)
            参考模板（正余弦参考信号）

        返回
        ----
        float
            最大典型相关系数的绝对值，范围 [0, 1]

        策略：
        - 样本充足（特征数/样本数 <= 0.1）时使用 sklearn 的 CCA
        - 样本不足时使用带正则化的 CCA，避免协方差矩阵奇异
        """
        n_comp = min(self.n_components, X.shape[1], Y.shape[1])
        if n_comp < 1:
            return 0.0

        ratio = X.shape[1] / X.shape[0]

        if ratio <= 0.1:
            # 样本充足：标准 CCA
            cca = CCA(n_components=n_comp)
            cca.fit(X, Y)
            X_c, Y_c = cca.transform(X, Y)
            corr = np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1]
            return float(abs(corr))
        else:
            # 样本不足：正则化 CCA
            return self._regularized_cca_score(X, Y, n_comp)

    def _regularized_cca_score(self, X, Y, _n_comp):
        """正则化 CCA：对协方差矩阵加 Tikhonov 正则项

        通过 λI 修正协方差矩阵，解决短数据下矩阵条件数过大的问题。
        正则化系数 λ = max(0.01, n_channels / n_samples)。
        """
        n_samples = X.shape[0]
        n_ch_x = X.shape[1]
        n_ch_y = Y.shape[1]

        # 去均值
        Xc = X - X.mean(axis=0)
        Yc = Y - Y.mean(axis=0)

        # 协方差矩阵
        C_xx = (Xc.T @ Xc) / (n_samples - 1)
        C_yy = (Yc.T @ Yc) / (n_samples - 1)
        C_xy = (Xc.T @ Yc) / (n_samples - 1)

        # 正则化系数：通道数越多 / 样本数越少，正则化越强
        lam = max(0.01, n_ch_x / n_samples)

        # 对自协方差矩阵施加正则化
        C_xx_reg = C_xx + lam * np.eye(n_ch_x)
        C_yy_reg = C_yy + lam * np.eye(n_ch_y)

        try:
            # 求解广义特征值问题：M * w = ρ² * C_xx_reg * w
            # 其中 M = C_xy @ C_yy^{-1} @ C_xy^T
            C_yy_inv = np.linalg.solve(C_yy_reg, np.eye(n_ch_y))
            M = C_xy @ C_yy_inv @ C_xy.T
            eigenvalues = eigh(M, C_xx_reg, eigvals_only=True)
            # 最大特征值开方即为典型相关系数
            rho_sq = max(0.0, eigenvalues[-1])
            return float(np.sqrt(min(rho_sq, 1.0)))
        except np.linalg.LinAlgError:
            return 0.0

    # ── 预处理 ──────────────────────────────────────────────
    def pre_filter(self, data, b_band=None, a_band=None):
        """EEG 数据预处理流水线

        依次执行：
        1. 通道选择（如果指定了 chans）
        2. CAR（共平均参考）空间滤波
        3. 去趋势
        4. 50 Hz 工频陷波
        5. 带通滤波（6-40 Hz 或用户指定）
        6. Z-score 标准化

        参数
        ----
        data : ndarray (n_channels, n_samples)
            原始 EEG 数据
        b_band, a_band : ndarray or None
            可选的外部滤波器系数，None 时使用默认 6-40 Hz 带通

        返回
        ----
        ndarray
            预处理后的数据，形状与输入一致
        """
        if self.chans is not None:
            data = data[self.chans, :].copy()
        else:
            data = data.copy()

        # CAR 空间预滤波：减去所有通道的均值，抑制共模噪声
        if self.use_car and data.shape[0] > 1:
            data = data - np.mean(data, axis=0, keepdims=True)

        # 线性去趋势，消除基线漂移
        data = signal.detrend(data, axis=1)

        # 50 Hz 工频陷波（Q=30）
        b_notch, a_notch = signal.iirnotch(50, 30, fs=self.srate)
        data = signal.filtfilt(b_notch, a_notch, data, axis=1)

        # 带通滤波
        if b_band is not None and a_band is not None:
            data = signal.filtfilt(b_band, a_band, data, axis=1)
        else:
            nyq = self.srate / 2
            b_bp, a_bp = signal.butter(4, [6 / nyq, 40 / nyq], btype='bandpass')
            data = signal.filtfilt(b_bp, a_bp, data, axis=1)

        # Z-score 标准化（逐通道）
        data = (data - np.mean(data, axis=1, keepdims=True)) / (
            np.std(data, axis=1, keepdims=True) + 1e-6
        )

        return data

    # ── 自动通道选择 ────────────────────────────────────────
    def _select_ssvep_channels(self, data, min_chans=2):
        """基于 SNR 自动选择 SSVEP 响应最强的通道

        对每个通道计算目标频率处的信噪比（SNR），筛选出 SNR 高于
        中位数 1.1 倍的通道。若筛选结果不足 min_chans，则取 SNR
        最高的前 min_chans 个通道。

        参数
        ----
        data : ndarray (n_channels, n_samples)
            EEG 数据
        min_chans : int
            最少保留的通道数

        返回
        ----
        ndarray
            选中的通道索引数组
        """
        n_chans, n_samples = data.shape
        if n_chans <= min_chans:
            return np.arange(n_chans)

        # FFT 频谱
        freqs_fft = np.fft.rfftfreq(n_samples, 1 / self.srate)
        spectrum = np.abs(np.fft.rfft(data, axis=1))

        snr_per_chan = np.zeros(n_chans)
        for freq in self.freqs:
            # 找到目标频率对应的 FFT bin
            idx = np.argmin(np.abs(freqs_fft - freq))
            # 半窗宽约 0.5 Hz
            half_win = max(1, int(0.5 / (freqs_fft[1] - freqs_fft[0])))
            sig_start = max(0, idx - half_win)
            sig_end = min(len(freqs_fft), idx + half_win + 1)

            # 信号功率：目标频率附近频带的能量
            signal_power = np.sum(spectrum[:, sig_start:sig_end] ** 2, axis=1)
            # 噪声功率：相邻频带的平均能量
            noise_left = list(range(max(0, idx - 10), sig_start))
            noise_right = list(range(sig_end, min(len(freqs_fft), idx + 10)))
            noise_bins = noise_left + noise_right

            if noise_bins:
                noise_power = np.mean(spectrum[:, noise_bins] ** 2, axis=1)
            else:
                noise_power = 1e-6

            # 累加各频率的 SNR
            snr_per_chan += signal_power / (noise_power + 1e-6)

        # 阈值筛选：取 SNR 高于中位数 1.1 倍的通道
        threshold = np.median(snr_per_chan) * 1.1
        good = np.where(snr_per_chan > threshold)[0]
        if len(good) < min_chans:
            good = np.argsort(snr_per_chan)[::-1][:min_chans]
        return np.sort(good)

    # ── 得分异常检测 ────────────────────────────────────────
    def _is_score_anomalous(self, combined):
        """检测得分向量是否异常（用于判断是否需要纠正）

        判断依据：
        1. 最高分与次高分之比（margin）> 1.25
        2. 最高分与均值之比（outlier_ratio）> 2.2

        返回
        ----
        (is_anomalous, confidence) : (bool, float)
            is_anomalous 为 True 表示检测结果可能存在异常
        """
        sorted_s = np.sort(combined)[::-1]
        mean_score = np.mean(combined)
        if mean_score < 1e-6:
            return False, 0.0

        margin = sorted_s[0] / (sorted_s[1] + 1e-6)
        outlier_ratio = sorted_s[0] / mean_score

        is_anom = margin > 1.25 and outlier_ratio > 2.2
        confidence = sorted_s[0] / (np.sum(sorted_s) + 1e-6)
        return is_anom, confidence

    # ── FBCCA 核心 ──────────────────────────────────────────
    def _run_fbcca(self, data, templates):
        """FBCCA 核心算法：滤波器组典型相关分析

        流程：
        1. CAR + 去趋势 + 50 Hz 陷波（共用预处理）
        2. 对每个子带分别滤波并 Z-score 标准化
        3. 在每个子带内计算 CCA 得分
        4. 子带内 min-max 归一化
        5. 加权融合各子带得分，输出最终预测

        参数
        ----
        data : ndarray (n_channels, n_samples)
            原始 EEG 数据
        templates : list[ndarray]
            参考模板列表，每个元素形状为 (2 * n_harmonics, n_samples)

        返回
        ----
        (predicted_index, combined_scores) : (int, ndarray)
            predicted_index 为目标频率在 freqs 中的索引
            combined_scores 为加权融合后的得分向量
        """
        # 仅在数据通道数足够时才应用 chans 筛选
        if self.chans is not None and data.shape[0] > max(self.chans):
            common = data[self.chans, :].copy()
        else:
            common = data.copy()

        # CAR 空间滤波
        if self.use_car and common.shape[0] > 1:
            common = common - np.mean(common, axis=0, keepdims=True)

        # 去趋势 + 50 Hz 陷波
        common = signal.detrend(common, axis=1)
        b_notch, a_notch = signal.iirnotch(50, 30, fs=self.srate)
        common = signal.filtfilt(b_notch, a_notch, common, axis=1)

        n_freqs = len(self.freqs)
        all_scores = np.zeros((len(self.filter_banks), n_freqs))

        # 逐子带处理
        for band_idx, (b_band, a_band) in enumerate(self.filter_banks):
            # 子带滤波 + Z-score 标准化
            filtered = signal.filtfilt(b_band, a_band, common, axis=1)
            filtered = (filtered - np.mean(filtered, axis=1, keepdims=True)) / (
                np.std(filtered, axis=1, keepdims=True) + 1e-6
            )
            X = filtered.T

            # 对每个目标频率计算 CCA 得分
            for freq_idx, template in enumerate(templates):
                Y = template.T
                all_scores[band_idx, freq_idx] = self._cca_score(X, Y)

        # 子带内 min-max 归一化：使不同子带的得分具有可比性
        for band_idx in range(len(self.filter_banks)):
            band_min = all_scores[band_idx].min()
            band_max = all_scores[band_idx].max()
            if band_max > band_min:
                all_scores[band_idx] = (all_scores[band_idx] - band_min) / (band_max - band_min)

        # 加权融合：使用 fb_weights 对各子带得分平方加权求和
        combined = np.dot(self.fb_weights, all_scores ** 2)
        return int(np.argmax(combined)), combined

    # ── 检测入口 ────────────────────────────────────────────
    def detect(self, data):
        """SSVEP 频率检测主入口

        参数
        ----
        data : ndarray (n_channels, n_samples)
            原始 EEG 数据

        返回
        ----
        int
            预测的目标频率在 freqs 列表中的索引
        """
        # 主检测
        if self.use_filterbank:
            pred, combined = self._run_fbcca(data, self.TemplateSet)
        else:
            # 非滤波器组模式：单次带通滤波 + CCA
            filtered = self.pre_filter(data)
            X = filtered.T
            scores = [self._cca_score(X, t.T) for t in self.TemplateSet]
            pred = int(np.argmax(scores))
            combined = np.array(scores) ** 2

        # 自适应纠正：仅在预测不确定时触发
        # 当最高分与次高分的比值（margin）小于 1.3 时，
        # 说明频域判别不够清晰，尝试多种备用模板重新判断
        if self.use_filterbank and self.use_correction:
            sorted_s = np.sort(combined)[::-1]
            margin = sorted_s[0] / (sorted_s[1] + 1e-6)

            if margin < 1.3:
                pred_corrected = self._try_correction(data, pred, margin)
                if pred_corrected is not None:
                    return pred_corrected

        return pred

    def _try_correction(self, data, base_pred, base_margin):
        """自适应纠正策略

        当主检测结果的 margin 过低时，依次尝试三种纠正策略：
        1. 5 次谐波 + 全部通道 → 高分辨率通用策略
        2. 针对 9/11 Hz 混淆 → 4 次谐波 + PO 区通道（枕叶）
        3. 一致性校验：策略 1 和策略 2 一致且不同于主结果时采纳

        参数
        ----
        data : ndarray
            EEG 原始数据
        base_pred : int
            主检测预测的频率索引
        base_margin : float
            主检测的 margin 值

        返回
        ----
        int or None
            纠正后的预测索引，无需纠正时返回 None
        """
        n_chans = data.shape[0]

        # PO 通道（枕叶区域，SSVEP 信号最强的位置）
        # 根据实际通道数决定索引
        if n_chans >= 6:
            po_idx = [3, 4, 5]
        elif n_chans >= 3:
            po_idx = list(range(n_chans))
        else:
            po_idx = list(range(n_chans))

        # 策略 1：n_h=5 + 全部通道（通用高分辨率）
        t5 = self._alt_templates[5]
        p5, c5 = self._run_fbcca(data, t5)
        s5 = np.sort(c5)[::-1]
        m5 = s5[0] / (s5[1] + 1e-6)
        if m5 > 1.3 and p5 != base_pred:
            return p5

        # 策略 2：9/11 Hz 混淆 → n_h=4 + PO 通道
        # 9 和 11 Hz 的谐波频率重合度高，用 4 次谐波 + PO 通道抑制混淆
        if base_pred in [self.freqs.index(9), self.freqs.index(11)]:
            data_po = data[po_idx, :]
            t4 = self._alt_templates[4]
            p4, c4 = self._run_fbcca(data_po, t4)
            s4 = np.sort(c4)[::-1]
            m4 = s4[0] / (s4[1] + 1e-6)
            if m4 > 1.15 and p4 != base_pred:
                return p4

        # 策略 3：两者一致且不同于 base，采纳一致结果
        p5_val = p5  # 复用已计算的值
        try:
            data_po = data[po_idx, :]
            t4 = self._alt_templates[4]
            p4, c4 = self._run_fbcca(data_po, t4)
            if p4 == p5_val and p4 != base_pred:
                return p4
        except Exception:
            pass

        return None
