import numpy as np
from scipy import signal
from scipy.linalg import eigh
from sklearn.cross_decomposition import CCA


class ssvepDetect:
    def __init__(self, srate, freqs, dataLen, n_harmonics=3, n_components=3,
                 chans=None, use_filterbank=True, use_car=True):
        self.srate = srate
        self.freqs = freqs
        self.n_harmonics = n_harmonics
        self.n_components = n_components
        self.chans = chans
        self.use_filterbank = use_filterbank
        self.use_car = use_car

        templLen = int(dataLen * srate)
        sample = np.linspace(0, (templLen - 1) / srate, templLen, endpoint=True)

        # 主模板
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

        if use_filterbank:
            self._init_filterbank()

    # ── 滤波器组 ────────────────────────────────────────────
    def _init_filterbank(self):
        nyq = self.srate / 2
        self.filter_banks = []
        for low_cut in [6, 10, 14, 18, 22, 26, 30, 34]:
            b, a = signal.butter(4, [low_cut / nyq, 40 / nyq], btype='bandpass')
            self.filter_banks.append((b, a))

        n_bands = len(self.filter_banks)
        a_param, b_param = 1.25, 0.25
        raw_weights = np.array(
            [n ** (-a_param) + b_param for n in range(1, n_bands + 1)]
        )
        self.fb_weights = raw_weights / raw_weights.sum()

    # ── CCA 得分 ────────────────────────────────────────────
    def _cca_score(self, X, Y):
        """混合 CCA：样本充足用 sklearn，样本不足用正则化"""
        n_comp = min(self.n_components, X.shape[1], Y.shape[1])
        if n_comp < 1:
            return 0.0

        ratio = X.shape[1] / X.shape[0]

        if ratio <= 0.1:
            cca = CCA(n_components=n_comp)
            cca.fit(X, Y)
            X_c, Y_c = cca.transform(X, Y)
            corr = np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1]
            return float(abs(corr))
        else:
            return self._regularized_cca_score(X, Y, n_comp)

    def _regularized_cca_score(self, X, Y, _n_comp):
        n_samples = X.shape[0]
        n_ch_x = X.shape[1]
        n_ch_y = Y.shape[1]

        Xc = X - X.mean(axis=0)
        Yc = Y - Y.mean(axis=0)

        C_xx = (Xc.T @ Xc) / (n_samples - 1)
        C_yy = (Yc.T @ Yc) / (n_samples - 1)
        C_xy = (Xc.T @ Yc) / (n_samples - 1)

        lam = max(0.01, n_ch_x / n_samples)

        C_xx_reg = C_xx + lam * np.eye(n_ch_x)
        C_yy_reg = C_yy + lam * np.eye(n_ch_y)

        try:
            C_yy_inv = np.linalg.solve(C_yy_reg, np.eye(n_ch_y))
            M = C_xy @ C_yy_inv @ C_xy.T
            eigenvalues = eigh(M, C_xx_reg, eigvals_only=True)
            rho_sq = max(0.0, eigenvalues[-1])
            return float(np.sqrt(min(rho_sq, 1.0)))
        except np.linalg.LinAlgError:
            return 0.0

    # ── 预处理 ──────────────────────────────────────────────
    def pre_filter(self, data, b_band=None, a_band=None):
        if self.chans is not None:
            data = data[self.chans, :].copy()
        else:
            data = data.copy()

        # CAR 空间预滤波
        if self.use_car and data.shape[0] > 1:
            data = data - np.mean(data, axis=0, keepdims=True)

        data = signal.detrend(data, axis=1)

        b_notch, a_notch = signal.iirnotch(50, 30, fs=self.srate)
        data = signal.filtfilt(b_notch, a_notch, data, axis=1)

        if b_band is not None and a_band is not None:
            data = signal.filtfilt(b_band, a_band, data, axis=1)
        else:
            nyq = self.srate / 2
            b_bp, a_bp = signal.butter(4, [6 / nyq, 40 / nyq], btype='bandpass')
            data = signal.filtfilt(b_bp, a_bp, data, axis=1)

        data = (data - np.mean(data, axis=1, keepdims=True)) / (
            np.std(data, axis=1, keepdims=True) + 1e-6
        )

        return data

    # ── 自动通道选择 ────────────────────────────────────────
    def _select_ssvep_channels(self, data, min_chans=2):
        n_chans, n_samples = data.shape
        if n_chans <= min_chans:
            return np.arange(n_chans)

        freqs_fft = np.fft.rfftfreq(n_samples, 1 / self.srate)
        spectrum = np.abs(np.fft.rfft(data, axis=1))

        snr_per_chan = np.zeros(n_chans)
        for freq in self.freqs:
            idx = np.argmin(np.abs(freqs_fft - freq))
            half_win = max(1, int(0.5 / (freqs_fft[1] - freqs_fft[0])))
            sig_start = max(0, idx - half_win)
            sig_end = min(len(freqs_fft), idx + half_win + 1)

            signal_power = np.sum(spectrum[:, sig_start:sig_end] ** 2, axis=1)
            noise_left = list(range(max(0, idx - 10), sig_start))
            noise_right = list(range(sig_end, min(len(freqs_fft), idx + 10)))
            noise_bins = noise_left + noise_right

            if noise_bins:
                noise_power = np.mean(spectrum[:, noise_bins] ** 2, axis=1)
            else:
                noise_power = 1e-6

            snr_per_chan += signal_power / (noise_power + 1e-6)

        threshold = np.median(snr_per_chan) * 1.1
        good = np.where(snr_per_chan > threshold)[0]
        if len(good) < min_chans:
            good = np.argsort(snr_per_chan)[::-1][:min_chans]
        return np.sort(good)

    # ── 得分异常检测 ────────────────────────────────────────
    def _is_score_anomalous(self, combined):
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
        if self.chans is not None:
            common = data[self.chans, :].copy()
        else:
            common = data.copy()

        if self.use_car and common.shape[0] > 1:
            common = common - np.mean(common, axis=0, keepdims=True)

        common = signal.detrend(common, axis=1)
        b_notch, a_notch = signal.iirnotch(50, 30, fs=self.srate)
        common = signal.filtfilt(b_notch, a_notch, common, axis=1)

        n_freqs = len(self.freqs)
        all_scores = np.zeros((len(self.filter_banks), n_freqs))

        for band_idx, (b_band, a_band) in enumerate(self.filter_banks):
            filtered = signal.filtfilt(b_band, a_band, common, axis=1)
            filtered = (filtered - np.mean(filtered, axis=1, keepdims=True)) / (
                np.std(filtered, axis=1, keepdims=True) + 1e-6
            )
            X = filtered.T

            for freq_idx, template in enumerate(templates):
                Y = template.T
                all_scores[band_idx, freq_idx] = self._cca_score(X, Y)

        # 子带内 min-max 归一化
        for band_idx in range(len(self.filter_banks)):
            band_min = all_scores[band_idx].min()
            band_max = all_scores[band_idx].max()
            if band_max > band_min:
                all_scores[band_idx] = (all_scores[band_idx] - band_min) / (band_max - band_min)

        combined = np.dot(self.fb_weights, all_scores ** 2)
        return int(np.argmax(combined)), combined

    # ── 检测入口 ────────────────────────────────────────────
    def detect(self, data):
        # 主检测
        if self.use_filterbank:
            pred, combined = self._run_fbcca(data, self.TemplateSet)
        else:
            filtered = self.pre_filter(data)
            X = filtered.T
            scores = [self._cca_score(X, t.T) for t in self.TemplateSet]
            pred = int(np.argmax(scores))
            combined = np.array(scores) ** 2

        # 自适应纠正：仅在预测不确定时触发
        if self.use_filterbank:
            sorted_s = np.sort(combined)[::-1]
            margin = sorted_s[0] / (sorted_s[1] + 1e-6)

            if margin < 1.3:
                pred_corrected = self._try_correction(data, pred, margin)
                if pred_corrected is not None:
                    return pred_corrected

        return pred

    def _try_correction(self, data, base_pred, base_margin):
        # 策略 1：n_h=5 + 全部通道（通用高分辨率）
        t5 = self._alt_templates[5]
        p5, c5 = self._run_fbcca(data, t5)
        s5 = np.sort(c5)[::-1]
        m5 = s5[0] / (s5[1] + 1e-6)
        if m5 > 1.3 and p5 != base_pred:
            return p5

        # 策略 2：9/11 Hz 混淆 → n_h=4 + PO 通道
        if base_pred in [self.freqs.index(9), self.freqs.index(11)]:
            po_chans = [3, 4, 5]
            data_po = data[po_chans, :]
            t4 = self._alt_templates[4]
            p4, c4 = self._run_fbcca(data_po, t4)
            s4 = np.sort(c4)[::-1]
            m4 = s4[0] / (s4[1] + 1e-6)
            if m4 > 1.15 and p4 != base_pred:
                return p4

        # 策略 3：两者一致且不同于 base，采纳
        p5_val = p5  # already computed
        try:
            po_chans = [3, 4, 5]
            data_po = data[po_chans, :]
            t4 = self._alt_templates[4]
            p4, c4 = self._run_fbcca(data_po, t4)
            if p4 == p5_val and p4 != base_pred:
                return p4
        except Exception:
            pass

        return None
