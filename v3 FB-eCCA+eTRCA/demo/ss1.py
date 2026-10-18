import numpy as np
from scipy import signal
from sklearn.cross_decomposition import CCA


class ssvepDetect:
    def __init__(self, srate, freqs, dataLen, n_harmonics=3, n_components=3,
                 chans=None, use_filterbank=True):
        self.srate = srate
        self.freqs = freqs
        self.n_harmonics = n_harmonics
        self.n_components = n_components
        self.chans = chans
        self.use_filterbank = use_filterbank

        templLen = int(dataLen * srate)
        sample = np.linspace(0, (templLen - 1) / srate, templLen, endpoint=True)

        self.TemplateSet = []
        for freq in freqs:
            tempset_list = []
            for h in range(1, n_harmonics + 1):
                phase = 2 * np.pi * freq * h * sample
                tempset_list.append(np.sin(phase))
                tempset_list.append(np.cos(phase))
            tempset = np.vstack(tempset_list)
            self.TemplateSet.append(tempset)

        # Backup templates with n_harmonics=4 for PO channel correction
        self.TemplateSet4 = []
        for freq in freqs:
            tempset_list = []
            for h in range(1, 5):
                phase = 2 * np.pi * freq * h * sample
                tempset_list.append(np.sin(phase))
                tempset_list.append(np.cos(phase))
            tempset = np.vstack(tempset_list)
            self.TemplateSet4.append(tempset)

        if use_filterbank:
            self._init_filterbank()

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

    def pre_filter(self, data, b_band=None, a_band=None):
        if self.chans is not None:
            data = data[self.chans, :].copy()
        else:
            data = data.copy()

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

    def _cca_score(self, X, Y):
        n_comp = min(self.n_components, X.shape[1], Y.shape[1])
        if n_comp < 1:
            return 0.0

        cca = CCA(n_components=n_comp)
        cca.fit(X, Y)
        X_c, Y_c = cca.transform(X, Y)

        corr = np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1]
        return float(abs(corr))

    def _run_fbcca(self, data, templates):
        """Run FBCCA pipeline, returns (prediction, combined_scores)"""
        if self.chans is not None:
            common = data[self.chans, :].copy()
        else:
            common = data.copy()

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

        for band_idx in range(len(self.filter_banks)):
            band_min = all_scores[band_idx].min()
            band_max = all_scores[band_idx].max()
            if band_max > band_min:
                all_scores[band_idx] = (all_scores[band_idx] - band_min) / (band_max - band_min)

        combined = np.dot(self.fb_weights, all_scores ** 2)
        return int(np.argmax(combined)), combined

    def detect(self, data):
        if self.use_filterbank:
            pred, combined = self._run_fbcca(data, self.TemplateSet)
        else:
            filtered = self.pre_filter(data)
            X = filtered.T
            scores = [self._cca_score(X, t.T) for t in self.TemplateSet]
            pred = int(np.argmax(scores))
            combined = np.array(scores)
            combined = combined ** 2  # match FBCCA squared convention

        # Correction: if primary (n_h=3, all chans) predicts 11 Hz with
        # high confidence, it may be a 9 Hz trial contaminated by alpha.
        # Re-evaluate with n_h=4 + PO channels for better harmonic separation.
        if self.use_filterbank and self.n_harmonics <= 3 and pred == 3:
            sorted_s = np.sort(combined)[::-1]
            margin = sorted_s[0] / (sorted_s[1] + 1e-6)
            if margin > 1.3:
                po_data = data[[3, 4, 5], :] if data.shape[0] > 5 else data
                if po_data.shape[0] >= 3:
                    backup_pred, _ = self._run_fbcca(po_data, self.TemplateSet4)
                    if backup_pred != pred:
                        return backup_pred

        return pred

    def _detect_single_band(self, data):
        filtered = self.pre_filter(data)
        X = filtered.T

        scores = [
            self._cca_score(X, template.T) for template in self.TemplateSet
        ]
        return int(np.argmax(scores))

    def _detect_filterbank(self, data):
        pred, _ = self._run_fbcca(data, self.TemplateSet)
        return pred
