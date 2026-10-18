import numpy as np
from scipy import signal
from sklearn.cross_decomposition import CCA

class ssvepDetect:
    def __init__(self, srate, freqs, dataLen, n_harmonics=3, n_components=3, chans=None):
        self.srate = srate
        self.freqs = freqs
        self.n_harmonics = n_harmonics
        self.n_components = n_components
        self.cca = CCA(n_components=n_components)
        self.TemplateSet = []
        self.chans = chans  # 选择通道索引，例如 [O1, O2, Oz, POz]

        templLen = int(dataLen * srate)
        sample = np.linspace(0, (templLen - 1) / srate, templLen, endpoint=True)

        # 构造高次谐波模板
        for freq in freqs:
            tempset_list = []
            for h in range(1, n_harmonics+1):
                _ = 2 * np.pi * freq * h * sample
                tempset_list.append(np.sin(_))
                tempset_list.append(np.cos(_))
            tempset = np.vstack(tempset_list)
            self.TemplateSet.append(tempset)

    def pre_filter(self, data):
        # 只选择指定通道
        if self.chans is not None:
            data = data[self.chans, :]

        # 去趋势
        data = signal.detrend(data, axis=1)
        # 标准化
        data = (data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-6)

        # 陷波 50Hz
        b_notch, a_notch = signal.iirnotch(50, 30, fs=self.srate)
        data = signal.filtfilt(b_notch, a_notch, data, axis=1)

        # 带通滤波 6-40Hz
        nyq = self.srate / 2
        b_bp, a_bp = signal.butter(4, [6/nyq, 40/nyq], btype='bandpass')
        data = signal.filtfilt(b_bp, a_bp, data, axis=1)

        return data

    def compute_channel_snr(self, data):
        # 简单 SNR: 信号方差 / 噪声方差
        snr = np.var(data, axis=1) / (np.var(data, axis=1).mean() + 1e-6)
        return snr / snr.sum()  # 归一化权重

    def detect(self, data):
        data = self.pre_filter(data)  # shape: chans x time
        cdata = data.T  # shape: time x chans

        # 通道加权
        weights = self.compute_channel_snr(data)  # shape: chans
        weighted_data = cdata * weights  # 每列乘权重

        p_scores = []
        for template in self.TemplateSet:
            ctemplate = template.T  # time x features
            self.cca.fit(weighted_data, ctemplate)
            X_c, Y_c = self.cca.transform(weighted_data, ctemplate)

            # 每个分量相关系数平方累加
            corrs = np.array([np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(self.n_components)])
            score = np.sum(corrs**2)
            p_scores.append(score)

        # 返回最可能的频率索引
        return p_scores.index(max(p_scores))