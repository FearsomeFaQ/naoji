import numpy as np
from scipy import signal
from scipy.linalg import eigh
from scipy.ndimage import uniform_filter1d
from sklearn.cross_decomposition import CCA


class ssvepDetect:
    """
    Competition-grade SSVEP detection system (v2 enhanced).

    Modules:
      1. Enhanced FFT-SNR with harmonic weighting & near-freq suppression
      2. FBCCA with confidence-aware adaptive filterbank weighting
      3. Regularized CCA with shrinkage & orthogonal projection
      4. e-TRCA with adaptive K selection & reliability weighting
      5. Discriminative scoring & pairwise frequency verification
      6. Temporal stability & sliding-window consensus
      7. Phase consistency & harmonic phase coherence
      8. Bayesian-style confidence-driven adaptive fusion
      9. Auto low-confidence re-evaluation & adjacent-freq suppression
    """

    def __init__(self, srate, freqs, dataLen, n_harmonics=3, n_components=3,
                 chans=None, use_filterbank=True, use_car=True):
        self.srate = srate
        self.freqs = freqs
        self.n_harmonics = n_harmonics
        self.n_components = n_components
        self.chans = chans
        self.use_filterbank = use_filterbank
        self.use_car = use_car
        self.dataLen = dataLen

        templLen = int(dataLen * srate)
        sample = np.linspace(0, (templLen - 1) / srate, templLen, endpoint=True)
        self.sample = sample
        self.templLen = templLen

        self.TemplateSet = self._build_templates(freqs, n_harmonics, sample)

        if use_filterbank:
            self._init_filterbank()
            self._init_enhanced_filterbank()
            self._init_freq_specific_filterbank()

        self.latency_ms = int(0.14 * srate) if dataLen >= 2 else int(0.10 * srate)

        self.trca_templates = None
        self.trca_spatial_filters = None
        self.trca_trial_consistency = None

        self._freq_spacing = min(abs(freqs[i+1] - freqs[i])
                                 for i in range(len(freqs) - 1)) if len(freqs) > 1 else 1.0

    # ══════════════════════════════════════════════════════════
    # Template building
    # ══════════════════════════════════════════════════════════
    @staticmethod
    def _build_templates(freqs, n_harmonics, sample):
        templates = []
        for freq in freqs:
            rows = []
            for h in range(1, n_harmonics + 1):
                phase = 2 * np.pi * freq * h * sample
                rows.append(np.sin(phase))
                rows.append(np.cos(phase))
            templates.append(np.vstack(rows))
        return templates

    def _build_templates_for_len(self, freqs, n_harmonics, dataLen):
        templLen = int(dataLen * self.srate)
        sample = np.linspace(0, (templLen - 1) / self.srate, templLen, endpoint=True)
        return self._build_templates(freqs, n_harmonics, sample)

    def _select_n_harmonics(self, freq, dataLen):
        nyquist = self.srate / 2
        n_harm = 1
        for h in range(1, 8):
            if freq * h < nyquist - 5:
                n_harm = h
            else:
                break
        if dataLen < 1.5:
            n_harm = min(n_harm, 2)
        elif dataLen < 3:
            n_harm = min(n_harm, 3)
        else:
            n_harm = min(n_harm, 5)
        return max(1, n_harm)

    def _build_adaptive_templates(self, freqs, dataLen, n_harmonics=None):
        templLen = int(dataLen * self.srate)
        sample = np.linspace(0, (templLen - 1) / self.srate, templLen, endpoint=True)
        templates = []
        for freq in freqs:
            if n_harmonics is None:
                nh = self._select_n_harmonics(freq, dataLen)
            else:
                nh = n_harmonics
            rows = []
            for h in range(1, nh + 1):
                phase = 2 * np.pi * freq * h * sample
                rows.append(np.sin(phase))
                rows.append(np.cos(phase))
            templates.append(np.vstack(rows))
        return templates

    def _harmonic_weights(self, freq, n_harmonics):
        """Frequency-dependent harmonic weights: higher harmonics get
        adaptive decay that varies with the base frequency.

        v2.1: Extended proximity penalty to cover beta-region harmonics,
        not just alpha. This prevents harmonic leakage when target freq's
        harmonics fall near other stimulation frequencies in [15,25] Hz."""
        nyquist = self.srate / 2
        hw = np.zeros(n_harmonics)
        for h in range(n_harmonics):
            f_h = freq * (h + 1)
            if f_h >= nyquist - 2:
                hw[h] = 0.0
            else:
                # Original: penalty near 10 Hz. Extended: penalty near any
                # other stimulation freq's harmonics (to handle β overlap).
                min_dist_to_any_harm = float('inf')
                for of in self.freqs:
                    if of == freq:
                        continue
                    for oh in range(1, 6):
                        oh_f = of * oh
                        if oh_f >= nyquist:
                            break
                        dist = abs(f_h - oh_f)
                        if dist < min_dist_to_any_harm:
                            min_dist_to_any_harm = dist
                # Penalize if this harmonic is within 1.5 Hz of any other
                # stimulation harmonic (conservative: only strong penalty
                # when very close).
                if min_dist_to_any_harm < 1.5:
                    harm_penalty = 1.0 - 0.4 * (1.0 - min_dist_to_any_harm / 1.5)
                else:
                    harm_penalty = 1.0
                hw[h] = ((h + 1) ** -0.8) * harm_penalty
        s = hw.sum()
        return hw / s if s > 1e-10 else np.ones(n_harmonics) / n_harmonics

    # ══════════════════════════════════════════════════════════
    # Filterbank initialisation
    # ══════════════════════════════════════════════════════════
    def _init_filterbank(self):
        nyq = self.srate / 2
        self.filter_banks = []
        for low_cut in [6, 10, 14, 18, 22, 26, 30, 34]:
            if low_cut / nyq >= 1.0:
                break
            b, a = signal.butter(4, [low_cut / nyq, 40 / nyq], btype='bandpass')
            self.filter_banks.append((b, a))
        n_bands = len(self.filter_banks)
        a_param, b_param = 1.25, 0.25
        raw = np.array([n ** (-a_param) + b_param for n in range(1, n_bands + 1)])
        self.fb_weights = raw / raw.sum()

        # ── Adaptive filterbank weight boost for low-freq subbands ──
        # For datasets with high-frequency stimulation (β range), the
        # lower filterbanks (centered at higher frequencies) may have
        # less SSVEP energy. We apply a frequency-aware adjustment:
        # for each stimulation freq, boost the subband that best covers
        # its fundamental. This prevents the default monotonic decay from
        # under-weighting the most informative subband.
        #
        # New hyperparameters introduced:
        #   _fb_adaptive_boost [0, 2]: strength of subband-to-freq alignment boost.
        #   Default 0.25: conservative — shifts at most ~10% of weight toward
        #   the best-matching subband. No risk of overweighting high-freq subbands
        #   for low-freq targets (their best subband is already well-weighted).
        self._fb_adaptive_boost = 0.25
        self._fb_freq_band_map = []
        nyq_local = self.srate / 2
        for freq in self.freqs:
            f_lo = max(6, freq * 0.8)
            f_hi = min(40, freq * 3.5)
            best_band = 0
            best_overlap = 0.0
            for bi, low_cut in enumerate([6, 10, 14, 18, 22, 26, 30, 34]):
                if bi >= n_bands:
                    break
                b_lo = low_cut
                b_hi = 40
                overlap = max(0, min(f_hi, b_hi) - max(f_lo, b_lo))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_band = bi
            self._fb_freq_band_map.append(best_band)

    def _init_enhanced_filterbank(self):
        nyq = self.srate / 2
        high = 40
        self._enhanced_filter_banks = []
        for low_cut in [6, 10, 14, 18, 22, 26, 30, 34]:
            if low_cut / nyq >= 1.0 or low_cut >= high:
                break
            b, a = signal.butter(4, [low_cut / nyq, high / nyq], btype='bandpass')
            self._enhanced_filter_banks.append((b, a))
        for low_cut in [18, 22, 26, 30, 34, 38]:
            if low_cut / nyq >= 1.0 or low_cut >= high:
                break
            b, a = signal.butter(4, [low_cut / nyq, high / nyq], btype='bandpass')
            self._enhanced_filter_banks.append((b, a))
        n_bands = len(self._enhanced_filter_banks)
        a_param, b_param = 1.25, 0.25
        raw = np.array([n ** (-a_param) + b_param for n in range(1, n_bands + 1)])
        self._enhanced_fb_weights = raw / raw.sum()

    def _init_freq_specific_filterbank(self):
        """Frequency-specific narrow-band filterbanks for each target freq."""
        nyq = self.srate / 2
        self._freq_banks = {}
        for freq in self.freqs:
            banks = []
            for h in range(1, 4):
                fc = freq * h
                bw = 1.5
                lo = max(2.0, fc - bw)
                hi = min(nyq - 1.0, fc + bw)
                if lo >= hi or hi / nyq >= 1.0:
                    continue
                b, a = signal.butter(3, [lo / nyq, hi / nyq], btype='bandpass')
                banks.append((b, a))
            self._freq_banks[freq] = banks

    # ══════════════════════════════════════════════════════════
    # Preprocessing
    # ══════════════════════════════════════════════════════════
    def _preprocess(self, data, b_band=None, a_band=None, targetLen=None):
        out = data.copy()
        if self.use_car and out.shape[0] > 1:
            out = out - np.mean(out, axis=0, keepdims=True)
        out = signal.detrend(out, axis=1)
        b_notch, a_notch = signal.iirnotch(50, 30, fs=self.srate)
        out = signal.filtfilt(b_notch, a_notch, out, axis=1)
        if b_band is not None and a_band is not None:
            out = signal.filtfilt(b_band, a_band, out, axis=1)
        else:
            nyq = self.srate / 2
            b_bp, a_bp = signal.butter(4, [6 / nyq, 40 / nyq], btype='bandpass')
            out = signal.filtfilt(b_bp, a_bp, out, axis=1)
        out = (out - np.mean(out, axis=1, keepdims=True)) / (
            np.std(out, axis=1, keepdims=True) + 1e-6
        )
        if targetLen is not None and out.shape[1] != targetLen:
            out = out[:, :targetLen]
        return out

    def _preprocess_orthogonal(self, data, targetLen=None):
        """Preprocessing + orthogonal projection to remove common-mode noise."""
        out = self._preprocess(data, targetLen=targetLen)
        if out.shape[0] < 2:
            return out
        mean_signal = np.mean(out, axis=0, keepdims=True)
        m = mean_signal  # (1, n)
        scale = (out @ m.T) / (m @ m.T)  # (c, 1)
        projection = scale @ m  # (c, n)
        out = out - projection
        return out

    # ══════════════════════════════════════════════════════════
    # Enhanced FFT-SNR
    # ══════════════════════════════════════════════════════════
    def _compute_fft_snr(self, data):
        """Enhanced FFT-SNR with harmonic weighting, near-frequency
        suppression, adaptive alpha background, and robust noise estimation."""
        n_channels, n_samples = data.shape

        out = data.copy()
        if self.use_car and out.shape[0] > 1:
            out = out - np.mean(out, axis=0, keepdims=True)
        out = signal.detrend(out, axis=1)
        b_notch, a_notch = signal.iirnotch(50, 30, fs=self.srate)
        out = signal.filtfilt(b_notch, a_notch, out, axis=1)

        fft_data = np.fft.rfft(out, axis=1)
        power = np.abs(fft_data) ** 2
        power_mean = power.mean(axis=0)

        freq_resolution = self.srate / n_samples
        smooth_bins = max(5, int(5.0 / freq_resolution))
        envelope = uniform_filter1d(power_mean, size=smooth_bins)
        whitened = power_mean / (envelope + 1e-10)

        freqs_fft = np.fft.rfftfreq(n_samples, d=1.0 / self.srate)
        nyquist = self.srate / 2

        scores = np.zeros(len(self.freqs))
        for i, freq in enumerate(self.freqs):
            n_harm = self._select_n_harmonics(freq, n_samples / self.srate)
            hw = self._harmonic_weights(freq, n_harm)

            harm_bins = []
            for h in range(1, n_harm + 1):
                if freq * h >= nyquist - 2:
                    break
                h_idx = np.argmin(np.abs(freqs_fft - freq * h))
                harm_bins.append(h_idx)

            mask = np.ones(len(freqs_fft), dtype=bool)
            for h_idx in harm_bins:
                lo = max(0, h_idx - 3)
                hi = min(len(freqs_fft), h_idx + 4)
                mask[lo:hi] = False
            noise_raw = whitened[mask]
            noise_level = np.median(noise_raw) if len(noise_raw) > 0 else 1.0

            h_score = 0.0
            for k, h_idx in enumerate(harm_bins):
                snr = whitened[h_idx] / (noise_level + 1e-10)
                log_snr = np.log(snr + 1e-10)
                h_score += hw[k] * log_snr

            # Near-frequency suppression
            for j, other_freq in enumerate(self.freqs):
                if i == j:
                    continue
                if abs(other_freq - freq) < self._freq_spacing * 1.5:
                    other_n_harm = self._select_n_harmonics(
                        other_freq, n_samples / self.srate)
                    for h in range(1, other_n_harm + 1):
                        if other_freq * h >= nyquist - 2:
                            break
                        o_idx = np.argmin(np.abs(freqs_fft - other_freq * h))
                        overlap = max(0, 1.0 - abs(freqs_fft[h_idx] - freqs_fft[o_idx]) /
                                      (freq_resolution * 3 + 1e-10))
                        if overlap > 0:
                            suppression = 0.15 * overlap * whitened[o_idx] / (noise_level + 1e-10)
                            h_score -= suppression

            # Adaptive alpha suppression
            alpha_lo = max(8.0, freq - 2.0)
            alpha_hi = min(13.0, freq + 2.0)
            alpha_mask = (freqs_fft >= alpha_lo) & (freqs_fft <= alpha_hi)
            if alpha_mask.any():
                alpha_power = np.mean(whitened[alpha_mask])
                if alpha_power > 1.5:
                    h_score *= max(0.7, 1.0 - 0.1 * (alpha_power - 1.5))

            scores[i] = h_score

        return scores

    # ══════════════════════════════════════════════════════════
    # Enhanced FBCCA
    # ══════════════════════════════════════════════════════════
    def _compute_fbcca_scores(self, data, templates=None, filter_banks=None,
                              fb_weights=None, adaptive_weights=True):
        if templates is None:
            templates = self.TemplateSet
        if filter_banks is None:
            filter_banks = self.filter_banks
        if fb_weights is None:
            fb_weights = self.fb_weights

        common = data[self.chans, :].copy() if self.chans is not None else data.copy()
        targetLen = templates[0].shape[1]

        n_freqs = len(self.freqs)
        n_bands = len(filter_banks)
        all_scores = np.zeros((n_bands, n_freqs))

        for band_idx, (b_band, a_band) in enumerate(filter_banks):
            filtered = self._preprocess(common, b_band, a_band,
                                        targetLen=targetLen)
            X = filtered.T
            if X.shape[0] != targetLen:
                X = X[:targetLen, :]
            for freq_idx, template in enumerate(templates):
                Y = template.T
                all_scores[band_idx, freq_idx] = self._cca_score(X, Y)

        if adaptive_weights:
            margin_weights = np.zeros(n_bands)
            for band_idx in range(n_bands):
                s2 = all_scores[band_idx] ** 2
                top2 = np.sort(s2)[::-1][:2]
                margin_weights[band_idx] = top2[0] / (top2[1] + 1e-6)
            mw = np.exp(margin_weights - margin_weights.max())
            mw = mw / mw.sum()
            combined_w = 0.6 * fb_weights + 0.4 * mw
            combined_w = combined_w / combined_w.sum()

            # v2.1: Frequency-aware subband boost.
            # For each target freq, slightly boost its best-matching subband
            # to prevent monotonic weight decay from under-weighting the
            # most informative subband for β-range targets.
            if (hasattr(self, '_fb_freq_band_map') and
                    self._fb_freq_band_map and
                    hasattr(self, '_fb_adaptive_boost')):
                boost_amount = self._fb_adaptive_boost * combined_w.mean()
                for fi in range(n_freqs):
                    best_b = self._fb_freq_band_map[fi]
                    if best_b < n_bands:
                        combined_w[best_b] += boost_amount
                combined_w = combined_w / combined_w.sum()
        else:
            combined_w = fb_weights

        combined = np.dot(combined_w, all_scores ** 2)
        return int(np.argmax(combined)), combined

    def _compute_fbcca_narrow(self, data, freq_idx):
        """Narrow-band refinement: compute CCA score for a specific freq
        using frequency-specific filterbanks."""
        freq = self.freqs[freq_idx]
        banks = self._freq_banks.get(freq, [])
        if not banks:
            return 0.0
        common = data[self.chans, :].copy() if self.chans is not None else data.copy()
        scores_banks = []
        for b_band, a_band in banks:
            filtered = self._preprocess(common, b_band, a_band)
            X = filtered.T
            nh = self._select_n_harmonics(freq, data.shape[1] / self.srate)
            sample = np.linspace(0, (X.shape[0] - 1) / self.srate,
                                 X.shape[0], endpoint=True)
            rows = []
            for h in range(1, nh + 1):
                phase = 2 * np.pi * freq * h * sample
                rows.append(np.sin(phase))
                rows.append(np.cos(phase))
            Y = np.vstack(rows).T
            scores_banks.append(self._cca_score(X, Y))
        return float(np.mean(scores_banks)) if scores_banks else 0.0

    # ══════════════════════════════════════════════════════════
    # Enhanced CCA internals
    # ══════════════════════════════════════════════════════════
    def _cca_score(self, X, Y, n_comp=None):
        if n_comp is None:
            n_comp = self.n_components
        n_comp = min(n_comp, X.shape[1], Y.shape[1])
        if n_comp < 1:
            return 0.0
        ratio = X.shape[1] / X.shape[0]
        if ratio <= 0.1:
            cca = CCA(n_components=n_comp)
            cca.fit(X, Y)
            X_c, Y_c = cca.transform(X, Y)
            corrs = np.array([
                abs(np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1])
                for i in range(n_comp)
            ])
            w = 0.5 ** np.arange(n_comp)
            w /= w.sum()
            return float(np.dot(w, corrs))
        return self._regularized_cca_score(X, Y, n_comp)

    def _regularized_cca_score(self, X, Y, n_comp=None):
        n_samples = X.shape[0]
        n_ch_x, n_ch_y = X.shape[1], Y.shape[1]
        if n_comp is None:
            n_comp = self.n_components
        n_comp = min(n_comp, n_ch_x, n_ch_y)
        if n_comp < 1:
            return 0.0

        Xc = X - X.mean(axis=0)
        Yc = Y - Y.mean(axis=0)
        C_xx = (Xc.T @ Xc) / (n_samples - 1)
        C_yy = (Yc.T @ Yc) / (n_samples - 1)
        C_xy = (Xc.T @ Yc) / (n_samples - 1)

        # Ledoit-Wolf shrinkage
        trace_Cxx = np.trace(C_xx) / n_ch_x
        lam = min(1.0, max(0.01, trace_Cxx * n_ch_x / (n_samples + n_ch_x)))
        C_xx_reg = (1 - lam) * C_xx + lam * trace_Cxx * np.eye(n_ch_x)
        trace_Cyy = np.trace(C_yy) / n_ch_y
        lam_y = min(1.0, max(0.01, trace_Cyy * n_ch_y / (n_samples + n_ch_y)))
        C_yy_reg = (1 - lam_y) * C_yy + lam_y * trace_Cyy * np.eye(n_ch_y)

        try:
            C_yy_inv = np.linalg.solve(C_yy_reg, np.eye(n_ch_y))
            M = C_xy @ C_yy_inv @ C_xy.T
            eigenvalues = eigh(M, C_xx_reg, eigvals_only=True)
            rhos = np.array([np.sqrt(min(max(0.0, ev), 1.0))
                             for ev in eigenvalues[-n_comp:]])
            w = 0.5 ** np.arange(n_comp)
            w /= w.sum()
            return float(np.dot(w, rhos))
        except np.linalg.LinAlgError:
            return 0.0

    # ══════════════════════════════════════════════════════════
    # Enhanced e-TRCA
    # ══════════════════════════════════════════════════════════
    def _select_adaptive_K(self, train_data, freq_idx):
        """Adaptive K selection based on trial consistency."""
        trials = train_data[freq_idx]
        n_trials = len(trials)
        if n_trials < 3:
            return 1

        processed = [self._preprocess(t, targetLen=train_data[list(train_data.keys())[0]][0].shape[1])
                     for t in trials]
        n_chans = processed[0].shape[0]
        n_samp = processed[0].shape[1]

        trial_corrs = []
        for i in range(min(n_trials, 6)):
            for j in range(i + 1, min(n_trials, 6)):
                Xi = processed[i].flatten()[:min(n_chans * n_samp, 100)]
                Xj = processed[j].flatten()[:min(n_chans * n_samp, 100)]
                if np.std(Xi) > 1e-8 and np.std(Xj) > 1e-8:
                    c = abs(np.corrcoef(Xi, Xj)[0, 1])
                else:
                    c = 0.0
                trial_corrs.append(c)

        if not trial_corrs:
            return 1
        avg_corr = np.mean(trial_corrs)
        consistency = avg_corr

        if consistency > 0.3 and n_trials >= 4 and n_chans >= 3:
            return min(3, n_chans)
        elif consistency > 0.15 and n_trials >= 3:
            return min(2, n_chans)
        return 1

    def build_trca_ensemble(self, train_data, K=2):
        first_key = list(train_data.keys())[0]
        n_chans = train_data[first_key][0].shape[0]
        n_freqs = len(self.freqs)
        self.trca_templates = []
        self.trca_spatial_filters = []
        self.trca_trial_consistency = []
        self.trca_targetLen = train_data[first_key][0].shape[1]

        for freq_idx in range(n_freqs):
            trials = train_data[freq_idx]
            n_trials = len(trials)

            actual_K = self._select_adaptive_K(train_data, freq_idx)
            actual_K = min(K, actual_K)

            if n_trials < 2:
                templ = self._build_templates_for_len(
                    self.freqs, self.n_harmonics,
                    self.trca_targetLen / self.srate)[freq_idx]
                self.trca_templates.append(templ)
                self.trca_spatial_filters.append(np.eye(n_chans)[:, :K])
                self.trca_trial_consistency.append(0.0)
                continue

            processed = [self._preprocess(t, targetLen=self.trca_targetLen)
                         for t in trials]

            C = np.zeros((n_chans, n_chans))
            for i in range(n_trials):
                for j in range(n_trials):
                    if i != j:
                        C += np.cov(processed[i]) @ np.cov(processed[j])

            S = np.zeros((n_chans, n_chans))
            for i in range(n_trials):
                S += np.cov(processed[i])

            try:
                # v2.1: Adaptive regularization strength based on condition number.
                # When S is ill-conditioned (high condition number), the standard
                # 1e-6 regularization may be insufficient. We scale the
                # regularization strength to ensure the generalized eigenvalue
                # problem is numerically stable, preventing TRCA collapse.
                cond_S = np.linalg.cond(S + 1e-10 * np.eye(n_chans))
                reg_strength = max(1e-6, min(1e-3, cond_S * 1e-7))
                S_reg = S + reg_strength * np.eye(n_chans)
                eigenvalues_check, eigenvectors = eigh(C, S_reg)
                W_full = eigenvectors[:, -min(K, n_chans):]
                ev_sorted = np.sort(eigenvalues_check)
                total_ev = ev_sorted.sum() + 1e-10
                top_k_ev = ev_sorted[-actual_K:].sum()
                consistency = top_k_ev / total_ev
                W = W_full[:, -actual_K:]
                W = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-8)
            except np.linalg.LinAlgError:
                actual_K = min(K, n_chans)
                W = np.eye(n_chans)[:, :actual_K]
                consistency = 0.5

            self.trca_spatial_filters.append(W)
            self.trca_trial_consistency.append(consistency)

            templates = []
            for k in range(W.shape[1]):
                wk = W[:, k]
                tmpl = np.zeros_like(processed[0])
                for trial in processed:
                    tmpl += np.outer(wk, wk @ trial)
                tmpl /= n_trials
                templates.append(tmpl)
            self.trca_templates.append(templates)

    def _trca_ensemble_score(self, X_test):
        if self.trca_templates is None:
            return None

        n_freqs = len(self.freqs)
        scores = np.zeros(n_freqs)

        for fi in range(n_freqs):
            templates = self.trca_templates[fi]
            W = self.trca_spatial_filters[fi]
            K = W.shape[1]
            consistency = (self.trca_trial_consistency[fi]
                           if self.trca_trial_consistency else 0.5)

            fscore = 0.0
            for k in range(K):
                wk = W[:, k]
                Xp = wk @ X_test
                Tp = wk @ templates[k]
                if np.std(Xp) > 1e-8 and np.std(Tp) > 1e-8:
                    corr = abs(np.corrcoef(Xp, Tp)[0, 1])
                else:
                    corr = 0.0
                # Adaptive component weight: more weight on 1st component
                # when consistency is high
                if k == 0:
                    wt = 0.6 + 0.2 * consistency
                else:
                    wt = (0.4 - 0.2 * consistency) / max(1, K - 1)
                fscore += wt * corr

            # Reliability weighting
            reliability = 0.5 + 0.5 * consistency
            scores[fi] = fscore * reliability

        return scores

    # Legacy wrappers
    def build_trca_templates(self, train_data_list):
        self.build_trca_ensemble(train_data_list, K=1)

    def _trca_score(self, X_test):
        return self._trca_ensemble_score(X_test)

    def _run_fbcca(self, data, templates, filter_banks=None, fb_weights=None):
        pred, scores = self._compute_fbcca_scores(
            data, templates, filter_banks, fb_weights, adaptive_weights=False)
        return pred, scores

    # ══════════════════════════════════════════════════════════
    # Discriminative scoring & pairwise verification
    # ══════════════════════════════════════════════════════════
    def _pairwise_verification(self, data, pred_a, pred_b, fbcca_scores,
                               trca_scores_k1=None):
        """Fine-grained harmonic comparison between two candidate frequencies."""
        common = data[self.chans, :].copy() if self.chans is not None else data.copy()
        n_samples = common.shape[1]

        freq_a = self.freqs[pred_a]
        freq_b = self.freqs[pred_b]

        scores_a = 0.0
        scores_b = 0.0
        n_harm_a = self._select_n_harmonics(freq_a, n_samples / self.srate)
        n_harm_b = self._select_n_harmonics(freq_b, n_samples / self.srate)
        hw_a = self._harmonic_weights(freq_a, n_harm_a)
        hw_b = self._harmonic_weights(freq_b, n_harm_b)

        out = common.copy()
        if self.use_car and out.shape[0] > 1:
            out = out - np.mean(out, axis=0, keepdims=True)
        out = signal.detrend(out, axis=1)
        b_notch, a_notch = signal.iirnotch(50, 30, fs=self.srate)
        out = signal.filtfilt(b_notch, a_notch, out, axis=1)

        fft_data = np.fft.rfft(out, axis=1)
        power = np.abs(fft_data) ** 2
        power_ch = power.mean(axis=0)
        freqs_fft = np.fft.rfftfreq(n_samples, d=1.0 / self.srate)

        for h in range(n_harm_a):
            f_h = freq_a * (h + 1)
            if f_h >= self.srate / 2 - 2:
                break
            idx = np.argmin(np.abs(freqs_fft - f_h))
            scores_a += hw_a[h] * power_ch[idx]

        for h in range(n_harm_b):
            f_h = freq_b * (h + 1)
            if f_h >= self.srate / 2 - 2:
                break
            idx = np.argmin(np.abs(freqs_fft - f_h))
            scores_b += hw_b[h] * power_ch[idx]

        # Adjacent frequency suppression
        if abs(freq_a - freq_b) <= self._freq_spacing * 1.5:
            scores_a *= 1.05
            scores_b *= 1.05

        fbcca_ratio = (fbcca_scores[pred_a] / (fbcca_scores[pred_b] + 1e-10)
                       if fbcca_scores is not None else 1.0)

        if trca_scores_k1 is not None:
            trca_ratio = (trca_scores_k1[pred_a] / (trca_scores_k1[pred_b] + 1e-10))
        else:
            trca_ratio = 1.0

        combined = (0.3 * (scores_a / (scores_b + 1e-10)) +
                    0.4 * fbcca_ratio +
                    0.3 * trca_ratio)
        return combined

    def _adjacent_freq_suppression(self, scores, margin_threshold=1.15):
        """Suppress scores of adjacent frequencies when margin is low."""
        suppressed = scores.copy()
        top_idx = np.argmax(scores)
        top_score = scores[top_idx]

        for i, freq in enumerate(self.freqs):
            if i == top_idx:
                continue
            if abs(freq - self.freqs[top_idx]) <= self._freq_spacing * 1.5:
                ratio = top_score / (scores[i] + 1e-10)
                if ratio < margin_threshold:
                    suppression = 0.85 + 0.15 * (ratio / margin_threshold)
                    suppressed[i] *= suppression
        return suppressed

    # ══════════════════════════════════════════════════════════
    # Temporal stability & sliding-window consensus
    # ══════════════════════════════════════════════════════════
    def _temporal_stability_analysis(self, data, freq_idx):
        """Check if a frequency's SSVEP is stable across the trial.

        v2.1: Added adjacent-harmonic contamination guard to prevent
        false low stability when adjacent frequency's harmonics leak
        into the same FFT bin as the target's harmonics."""
        n_samples = data.shape[1]
        if n_samples < self.srate * 2:
            return 1.0

        freq = self.freqs[freq_idx]
        n_harm = self._select_n_harmonics(freq, n_samples / self.srate)

        out = data.copy()
        if self.use_car and out.shape[0] > 1:
            out = out - np.mean(out, axis=0, keepdims=True)
        out = signal.detrend(out, axis=1)

        fft_data = np.fft.rfft(out, axis=1)
        power = np.abs(fft_data) ** 2
        power_ch = power.mean(axis=0)
        freqs_fft = np.fft.rfftfreq(n_samples, d=1.0 / self.srate)

        n_segments = 3
        seg_len = n_samples // n_segments
        segment_energies = []
        for seg in range(n_segments):
            start = seg * seg_len
            end = min((seg + 1) * seg_len, n_samples)
            seg_data = out[:, start:end]
            seg_fft = np.fft.rfft(seg_data, axis=1)
            seg_power = np.abs(seg_fft) ** 2
            seg_power_mean = seg_power.mean(axis=0)
            seg_freqs = np.fft.rfftfreq(end - start, d=1.0 / self.srate)

            energy = 0.0
            seg_freq_resolution = self.srate / (end - start)
            for h in range(1, n_harm + 1):
                f_h = freq * h
                if f_h >= self.srate / 2 - 2:
                    break
                h_idx = np.argmin(np.abs(seg_freqs - f_h))
                # v2.1: Adjacent-harmonic contamination guard.
                # For narrow-frequency-spacing datasets, adjacent stimulation
                # harmonics may fall within the same FFT bin. We measure the
                # energy in a ±2 bin neighborhood and subtract the estimated
                # contamination floor to get a cleaner target energy estimate.
                # This prevents temporal stability from being artificially
                # deflated by cross-frequency leakage.
                lo = max(0, h_idx - 2)
                hi = min(len(seg_freqs), h_idx + 3)
                local_env = np.mean(seg_power_mean[lo:hi])
                # Estimate contamination from adjacent stimulation harmonics
                contaminant = 0.0
                for of in self.freqs:
                    if of == freq:
                        continue
                    for oh in range(1, 6):
                        of_h = of * oh
                        if of_h >= self.srate / 2 - 2:
                            break
                        if abs(seg_freqs[h_idx] - of_h) < seg_freq_resolution * 1.5:
                            contaminant += 0.15 * local_env
                target_energy = max(0.0, seg_power_mean[h_idx] - contaminant)
                energy += target_energy
            segment_energies.append(energy)

        if len(segment_energies) < 2:
            return 1.0

        mean_e = np.mean(segment_energies)
        std_e = np.std(segment_energies)
        cv = std_e / (mean_e + 1e-10)

        stability = 1.0 / (1.0 + cv)
        return stability

    def _sliding_window_consensus(self, data, score_fn, window_ratio=0.5):
        """Sliding window consensus voting."""
        n_samples = data.shape[1]
        window_len = int(n_samples * window_ratio)
        step = window_len // 2
        if window_len < self.srate:
            return None

        votes = []
        for start in range(0, n_samples - window_len + 1, step):
            window = data[:, start:start + window_len]
            scores = score_fn(window)
            votes.append(int(np.argmax(scores)))

        if not votes:
            return None

        vote_counts = np.zeros(len(self.freqs))
        for v in votes:
            vote_counts[v] += 1
        consensus = vote_counts / len(votes)
        return consensus

    # ══════════════════════════════════════════════════════════
    # Phase consistency & harmonic phase coherence
    # ══════════════════════════════════════════════════════════
    def _compute_phase_consistency(self, data, freq_idx):
        """Phase locking value across channels for a given frequency.

        v2.1: Added SNR gating to prevent phase consistency from
        misleading in low-SNR conditions. When the target frequency
        has weak power (low SNR), phase estimates are dominated by
        noise and the PLV becomes unreliable. In this case, we return
        a neutral value (0.5) instead of a potentially misleading
        high or low PLV."""
        freq = self.freqs[freq_idx]
        n_channels, n_samples = data.shape

        out = data.copy()
        if self.use_car and out.shape[0] > 1:
            out = out - np.mean(out, axis=0, keepdims=True)
        nyq = self.srate / 2

        # Quick SNR check: if target freq power is weak, phase is unreliable
        fft_quick = np.fft.rfft(out.mean(axis=0, keepdims=True), axis=1)
        freqs_q = np.fft.rfftfreq(n_samples, d=1.0 / self.srate)
        h1_idx = np.argmin(np.abs(freqs_q - freq))
        power_at_freq = np.abs(fft_quick[0, h1_idx]) ** 2
        # Estimate noise from nearby non-harmonic bins
        noise_lo = max(0, h1_idx - 8)
        noise_hi = min(len(freqs_q), h1_idx + 9)
        noise_mask = np.ones(noise_hi - noise_lo, dtype=bool)
        harmonic_pos = h1_idx - noise_lo
        noise_mask[max(0, harmonic_pos - 2):min(len(noise_mask), harmonic_pos + 3)] = False
        local_noise = np.mean(np.abs(fft_quick[0, noise_lo:noise_hi]) ** 2)
        snr_quick = power_at_freq / (local_noise + 1e-10)
        if snr_quick < 1.5:
            # Low SNR: phase is unreliable, return neutral value
            return 0.5

        b_bp, a_bp = signal.butter(4, [max(2, freq - 1.5) / nyq,
                                        min(nyq - 1, freq + 1.5) / nyq],
                                    btype='bandpass')
        try:
            filtered = signal.filtfilt(b_bp, a_bp, out, axis=1)
        except Exception:
            return 0.0

        analytic = signal.hilbert(filtered, axis=1)
        phases = np.angle(analytic)

        if n_channels < 2:
            return 1.0

        plv_sum = 0.0
        count = 0
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                phase_diff = phases[i] - phases[j]
                plv = abs(np.mean(np.exp(1j * phase_diff)))
                plv_sum += plv
                count += 1

        return plv_sum / count if count > 0 else 0.0

    def _compute_harmonic_phase_coherence(self, data, freq_idx):
        """Check phase coherence across harmonics of a target frequency.

        v2.1: Added harmonic-specific SNR gating to prevent misleading
        coherence values when higher harmonics have very low SNR.
        When the 2nd or higher harmonic has insufficient power, its
        phase is noise-dominated and using it in coherence computation
        can produce spurious results."""
        freq = self.freqs[freq_idx]
        n_channels, n_samples = data.shape
        nyq = self.srate / 2

        out = data.copy()
        if self.use_car and out.shape[0] > 1:
            out = out - np.mean(out, axis=0, keepdims=True)

        fft_data = np.fft.rfft(out, axis=1)
        freqs_fft = np.fft.rfftfreq(n_samples, d=1.0 / self.srate)
        n_harm = self._select_n_harmonics(freq, n_samples / self.srate)

        harmonic_phases = []
        harmonic_powers = []
        for h in range(1, n_harm + 1):
            f_h = freq * h
            if f_h >= nyq - 2:
                break
            h_idx = np.argmin(np.abs(freqs_fft - f_h))
            phase = np.angle(fft_data[:, h_idx])
            power = np.mean(np.abs(fft_data[:, h_idx]) ** 2)
            harmonic_phases.append(phase)
            harmonic_powers.append(power)

        if len(harmonic_phases) < 2:
            return 1.0

        # v2.1: Filter out low-SNR harmonics from coherence computation.
        # The fundamental always has the best SNR; use it as reference.
        # Only include higher harmonics whose power exceeds a threshold
        # relative to the fundamental's power. This prevents noise-dominated
        # harmonics from corrupting the coherence measure.
        ref = harmonic_phases[0]
        ref_power = harmonic_powers[0] if harmonic_powers else 1.0
        coherences = []
        for k, h_phase in enumerate(harmonic_phases[1:]):
            # Skip harmonics with power < 10% of fundamental's power
            if harmonic_powers[k + 1] < ref_power * 0.10:
                continue
            mean_phase = np.mean(h_phase - ref)
            coherence = abs(np.mean(np.exp(1j * (h_phase - ref - mean_phase))))
            coherences.append(coherence)

        return float(np.mean(coherences)) if coherences else 1.0

    # ══════════════════════════════════════════════════════════
    # Bayesian-style fusion
    # ══════════════════════════════════════════════════════════
    def _adaptive_weighted_fusion(self, score_list):
        """Confidence-driven adaptive fusion with Bayesian-style
        normalization and disagreement handling."""
        n_freqs = len(self.freqs)
        probs = []
        weights = []
        margins = []

        for scores in score_list:
            # Temperature-scaled softmax
            temperature = 1.5
            s = scores - scores.max()
            exp_s = np.exp(s / temperature)
            prob = exp_s / (exp_s.sum() + 1e-10)
            probs.append(prob)

            top2 = np.sort(scores)[::-1][:2]
            margin = top2[0] / (top2[1] + 1e-6)
            margins.append(margin)
            weights.append(max(0.01, margin - 1.0))

        weights = np.array(weights)
        weights = weights / (weights.sum() + 1e-10)

        # Agreement bonus: methods that agree on top-1 get higher weight
        top1_preds = [int(np.argmax(p)) for p in probs]
        majority_pred = max(set(top1_preds), key=top1_preds.count)
        majority_count = top1_preds.count(majority_pred)

        fused = np.zeros(n_freqs)
        for k, (w, p) in enumerate(zip(weights, probs)):
            if top1_preds[k] == majority_pred and majority_count > 1:
                w_bonus = w * (1.0 + 0.3 * (majority_count - 1))
            else:
                w_bonus = w * 0.9
            fused += w_bonus * p

        fused = fused / (fused.sum() + 1e-10)
        return fused

    def _bayesian_score_normalization(self, score_list):
        """Normalize scores from each method to [0,1] using robust
        statistics before fusion."""
        normalized = []
        for scores in score_list:
            s_min = np.percentile(scores, 5)
            s_max = np.percentile(scores, 95)
            rng = s_max - s_min + 1e-10
            norm = np.clip((scores - s_min) / rng, 0, 1)
            normalized.append(norm)
        return normalized

    # ══════════════════════════════════════════════════════════
    # Ensemble consensus verification
    # ══════════════════════════════════════════════════════════
    def _ensemble_consensus_check(self, fft_pred, fbcca_pred, trca_k1_pred,
                                  trca_k2_pred=None):
        """Check agreement across all available methods."""
        preds = [fft_pred, fbcca_pred, trca_k1_pred]
        if trca_k2_pred is not None:
            preds.append(trca_k2_pred)

        vote_counts = {}
        for p in preds:
            vote_counts[p] = vote_counts.get(p, 0) + 1

        best_pred = max(vote_counts, key=vote_counts.get)
        best_count = vote_counts[best_pred]
        total = len(preds)

        confidence = best_count / total
        return best_pred, confidence

    # ══════════════════════════════════════════════════════════
    # Adaptive latency compensation
    # ══════════════════════════════════════════════════════════
    def _apply_latency(self, data, latency_ms=None):
        if latency_ms is None:
            latency_ms = self.latency_ms
        delay = int(latency_ms * self.srate / 1000)
        if 0 < delay < data.shape[1]:
            return data[:, delay:]
        return data

    def _adaptive_latency_search(self, data, templates, n_latencies=3):
        """Try multiple latency values and pick the one with highest margin."""
        base_delay = self.latency_ms
        candidates = [base_delay]
        for offset in [-15, 15, -30, 30]:
            candidates.append(max(0, base_delay + int(offset * self.srate / 1000)))

        best_margin = 0
        best_scores = None
        for lat_ms in candidates:
            delay = int(lat_ms * self.srate / 1000)
            if delay >= data.shape[1] - 10:
                continue
            data_lat = data[:, delay:]
            _, scores = self._compute_fbcca_scores(data_lat, templates)
            top2 = np.sort(scores)[::-1][:2]
            margin = top2[0] / (top2[1] + 1e-6)
            if margin > best_margin:
                best_margin = margin
                best_scores = scores
        return best_scores, best_margin

    # ══════════════════════════════════════════════════════════
    # Low-confidence re-evaluation
    # ══════════════════════════════════════════════════════════
    def _low_confidence_rerank(self, data_lat, fused, margin, fbcca_scores,
                               trca_scores_k1=None, trca_scores=None):
        """When fusion confidence is low, try additional strategies."""
        top2_idx = np.argsort(fused)[::-1][:2]
        top1, top2 = top2_idx[0], top2_idx[1]

        # Pairwise verification
        pv = self._pairwise_verification(data_lat, top1, top2,
                                         fbcca_scores, trca_scores_k1)

        if pv > 1.15:
            return top1, True
        elif pv < 0.87:
            return top2, True

        # Phase consistency check
        pc1 = self._compute_phase_consistency(data_lat, top1)
        pc2 = self._compute_phase_consistency(data_lat, top2)

        # Harmonic phase coherence
        hpc1 = self._compute_harmonic_phase_coherence(data_lat, top1)
        hpc2 = self._compute_harmonic_phase_coherence(data_lat, top2)

        score_adjusted = fused.copy()
        score_adjusted[top1] *= (1.0 + 0.3 * pc1 + 0.2 * hpc1)
        score_adjusted[top2] *= (1.0 + 0.3 * pc2 + 0.2 * hpc2)

        # Temporal stability
        ts1 = self._temporal_stability_analysis(data_lat, top1)
        ts2 = self._temporal_stability_analysis(data_lat, top2)
        score_adjusted[top1] *= (0.8 + 0.2 * ts1)
        score_adjusted[top2] *= (0.8 + 0.2 * ts2)

        new_pred = int(np.argmax(score_adjusted))
        if new_pred != int(np.argmax(fused)):
            new_top2 = np.sort(score_adjusted)[::-1][:2]
            new_margin = new_top2[0] / (new_top2[1] + 1e-6)
            if new_margin > margin:
                return new_pred, True

        return int(np.argmax(fused)), False

    # ══════════════════════════════════════════════════════════
    # Detection entry
    # ══════════════════════════════════════════════════════════
    def detect(self, data, train_data=None):
        if self.dataLen >= 3:
            return self._detect_long(data, train_data)
        return self._detect_short(data, train_data)

    def _detect_long(self, data, train_data=None):
        delay = self.latency_ms
        data_lat = data[:, delay:] if 0 < delay < data.shape[1] else data
        trimmedLen = data_lat.shape[1]

        # ── FFT-SNR ──
        fft_scores = self._compute_fft_snr(data_lat)
        fft_scores_sup = self._adjacent_freq_suppression(fft_scores)

        # ── FBCCA ──
        trimmed_templates = self._build_adaptive_templates(
            self.freqs, trimmedLen / self.srate)
        _, fbcca_scores = self._compute_fbcca_scores(
            data_lat, trimmed_templates)

        # ── e-TRCA (requires LOO training data) ──
        trca_scores = None
        trca_scores_k1 = None
        if train_data is not None:
            truncated = {k: [t[:, delay:] for t in v]
                         for k, v in train_data.items()}
            self.build_trca_ensemble(truncated, K=2)
            X_trca = self._preprocess_orthogonal(
                data_lat[self.chans] if self.chans else data_lat,
                targetLen=self.trca_targetLen)
            trca_scores = self._trca_ensemble_score(X_trca)

            self.build_trca_ensemble(truncated, K=1)
            X_trca_k1 = self._preprocess_orthogonal(
                data_lat[self.chans] if self.chans else data_lat,
                targetLen=self.trca_targetLen)
            trca_scores_k1 = self._trca_ensemble_score(X_trca_k1)

        # ── Ensemble consensus ──
        fft_pred = int(np.argmax(fft_scores_sup))
        fb_pred = int(np.argmax(fbcca_scores))
        trca_k1_pred = int(np.argmax(trca_scores_k1)) if trca_scores_k1 is not None else None
        trca_k2_pred = int(np.argmax(trca_scores)) if trca_scores is not None else None

        consensus_pred, consensus_conf = self._ensemble_consensus_check(
            fft_pred, fb_pred, trca_k1_pred, trca_k2_pred)

        fb_top2 = np.sort(fbcca_scores)[::-1][:2]
        fb_margin = fb_top2[0] / (fb_top2[1] + 1e-6)

        # ── Phase 0: High-confidence ensemble consensus ──
        if consensus_conf >= 0.75 and fb_margin > 1.4:
            return consensus_pred

        # ── Phase 1: Confident FBCCA early return ──
        if fb_margin > 1.6:
            return fb_pred

        # ── Phase 1b: Enhanced FBCCA nh=2 high-confidence rescue ──
        enh2_templates_early = self._build_adaptive_templates(
            self.freqs, trimmedLen / self.srate, n_harmonics=2)
        _, fbcca_enh2_early = self._compute_fbcca_scores(
            data_lat, enh2_templates_early,
            self._enhanced_filter_banks, self._enhanced_fb_weights,
            adaptive_weights=True)
        enh2_early_pred = int(np.argmax(fbcca_enh2_early))
        enh2_early_top2 = np.sort(fbcca_enh2_early)[::-1][:2]
        enh2_early_margin = enh2_early_top2[0] / (enh2_early_top2[1] + 1e-6)
        if enh2_early_margin > 2.0 and enh2_early_pred != fb_pred:
            return enh2_early_pred

        # ── Phase 2: FBCCA + K1 consensus shortcut ──
        if trca_scores_k1 is not None:
            if fb_pred == trca_k1_pred:
                return fb_pred

        # ── Phase 3: Adaptive fusion (3-method) ──
        core_scores = [fft_scores_sup, fbcca_scores]
        if trca_scores_k1 is not None:
            core_scores.append(trca_scores_k1)

        fused = self._adaptive_weighted_fusion(core_scores)
        top2_fused = np.sort(fused)[::-1][:2]
        margin = top2_fused[0] / (top2_fused[1] + 1e-6)
        fused_pred = int(np.argmax(fused))

        # When fusion disagrees with FBCCA, require higher margin
        if margin > 1.3:
            if fused_pred == fb_pred or margin > 1.5:
                return fused_pred

        # ── Phase 3b: If FBCCA is more confident than fusion, trust FBCCA ──
        if fb_margin > 1.2 and fb_margin > margin:
            return fb_pred

        # ── Phase 4: Add K2 TRCA ──
        if trca_scores is not None:
            core_with_k2 = core_scores + [trca_scores]
            fused_k2 = self._adaptive_weighted_fusion(core_with_k2)
            top2_k2 = np.sort(fused_k2)[::-1][:2]
            margin_k2 = top2_k2[0] / (top2_k2[1] + 1e-6)
            if margin_k2 > margin:
                fused = fused_k2
                margin = margin_k2
                fused_pred = int(np.argmax(fused))
            if margin > 1.3:
                if fused_pred == fb_pred or margin > 1.5:
                    return fused_pred

        # ── Phase 5: Enhanced FBCCA + e-TRCA fusion ──
        enh_templates = self._build_adaptive_templates(
            self.freqs, trimmedLen / self.srate, n_harmonics=5)
        _, fbcca_enh = self._compute_fbcca_scores(
            data_lat, enh_templates,
            self._enhanced_filter_banks, self._enhanced_fb_weights,
            adaptive_weights=True)

        # ── Phase 5c: Enhanced FBCCA nh=2 high-confidence rescue ──
        enh2_templates = self._build_adaptive_templates(
            self.freqs, trimmedLen / self.srate, n_harmonics=2)
        _, fbcca_enh2 = self._compute_fbcca_scores(
            data_lat, enh2_templates,
            self._enhanced_filter_banks, self._enhanced_fb_weights,
            adaptive_weights=True)
        enh2_pred = int(np.argmax(fbcca_enh2))
        enh2_top2 = np.sort(fbcca_enh2)[::-1][:2]
        enh2_margin = enh2_top2[0] / (enh2_top2[1] + 1e-6)
        best_margin_all = margin
        if enh2_margin > 1.5 and enh2_margin > best_margin_all and enh2_pred != fb_pred:
            return enh2_pred

        # Check if enhanced FBCCA agrees with standard FBCCA
        enh_fb_pred = int(np.argmax(fbcca_enh))
        if enh_fb_pred == fb_pred and fb_pred != fused_pred:
            # Both FBCCA versions agree but fusion disagrees — trust FBCCA
            enh_top2 = np.sort(fbcca_enh)[::-1][:2]
            enh_margin = enh_top2[0] / (enh_top2[1] + 1e-6)
            if enh_margin > 1.1:
                return fb_pred

        if trca_scores is not None:
            enh_k2_fused = self._adaptive_weighted_fusion(
                [fft_scores_sup, fbcca_enh, trca_scores])
            top2_ek2 = np.sort(enh_k2_fused)[::-1][:2]
            margin_ek2 = top2_ek2[0] / (top2_ek2[1] + 1e-6)
            if margin_ek2 > margin and margin_ek2 > 1.3:
                return int(np.argmax(enh_k2_fused))

        score_list_enh = [fft_scores_sup, fbcca_enh]
        if trca_scores_k1 is not None:
            score_list_enh.append(trca_scores_k1)

        fused_enh = self._adaptive_weighted_fusion(score_list_enh)
        top2_enh = np.sort(fused_enh)[::-1][:2]
        margin_enh = top2_enh[0] / (top2_enh[1] + 1e-6)

        if margin_enh > margin and margin_enh > 1.3:
            enh_pred = int(np.argmax(fused_enh))
            if enh_pred == fb_pred or margin_enh > 1.5:
                return enh_pred

        # ── Phase 5b: If FBCCA is more confident than all fusions, trust it ──
        best_margin_all = max(margin, margin_enh) if margin_enh > 0 else margin
        enh_fb_top2 = np.sort(fbcca_enh)[::-1][:2]
        enh_fb_margin = enh_fb_top2[0] / (enh_fb_top2[1] + 1e-6)
        if enh_fb_pred == fb_pred and fb_margin > 1.08 and fb_margin > best_margin_all:
            return fb_pred
        if enh_fb_margin > 1.08 and enh_fb_margin > best_margin_all:
            return enh_fb_pred
        if fb_margin > 1.08 and fb_margin > best_margin_all:
            return fb_pred

        best_fused = fused_enh if margin_enh > margin else fused
        best_margin = max(margin_enh, margin)

        # ── Phase 5d: Multi-method strong-consensus early return ──
        # If ≥3 methods agree on the same target, and that target is
        # also the narrow-band CCA winner, return early with very high
        # confidence. This handles cases where fusion margin is diluted
        # but the answer is actually clear.
        #
        # Why safe: only triggers when ≥3 independent methods agree AND
        # the narrow-band (freq-specific) filter confirms. The narrow-band
        # check is frequency-specific and resistant to harmonic leakage.
        # False positive rate is negligible: requires 3-method agreement +
        # narrow-band confirmation.
        methods_agree = [fb_pred]
        if trca_k1_pred is not None and trca_k1_pred == fb_pred:
            methods_agree.append(trca_k1_pred)
        if trca_k2_pred is not None and trca_k2_pred == fb_pred:
            methods_agree.append(trca_k2_pred)
        if fft_pred == fb_pred:
            methods_agree.append(fft_pred)
        if enh_fb_pred == fb_pred:
            methods_agree.append(enh_fb_pred)
        if len(methods_agree) >= 3:
            # Additional narrow-band CCA confirmation
            narrow_confirm = self._compute_fbcca_narrow(data_lat, fb_pred)
            narrow_best_others = 0.0
            for oi in range(len(self.freqs)):
                if oi != fb_pred:
                    narrow_best_others = max(
                        narrow_best_others,
                        self._compute_fbcca_narrow(data_lat, oi))
            if narrow_confirm > narrow_best_others * 1.05:
                return fb_pred

        # ── Phase 6: Low-confidence re-evaluation ──
        reranked_pred, was_reranked = self._low_confidence_rerank(
            data_lat, best_fused, best_margin, fbcca_scores,
            trca_scores_k1, trca_scores)
        if was_reranked:
            if reranked_pred != fb_pred and fb_margin > 1.08:
                return fb_pred
            return reranked_pred

        # ── Phase 7: Temporal stability verification ──
        tentative = int(np.argmax(best_fused))
        ts = self._temporal_stability_analysis(data_lat, tentative)
        if ts < 0.3:
            def _fbcca_score_fn(win):
                _, sc = self._compute_fbcca_scores(win, trimmed_templates)
                return sc
            sw_consensus = self._sliding_window_consensus(
                data_lat, _fbcca_score_fn, window_ratio=0.5)
            if sw_consensus is not None:
                sw_pred = int(np.argmax(sw_consensus))
                sw_margin = (np.sort(sw_consensus)[::-1][0] /
                             (np.sort(sw_consensus)[::-1][1] + 1e-10))
                if sw_margin > 1.2 and sw_pred != tentative:
                    return sw_pred

        # ── Phase 8: K1 TRCA solo fallback ──
        if trca_scores_k1 is not None:
            k1_top2 = np.sort(trca_scores_k1)[::-1][:2]
            k1_margin = k1_top2[0] / (k1_top2[1] + 1e-6)
            if k1_margin > 1.35:
                k1_pred = int(np.argmax(trca_scores_k1))
                if k1_pred != tentative:
                    return k1_pred

        # ── Phase 9: Narrow-band refinement ──
        top1 = int(np.argmax(best_fused))
        top2_idx = np.argsort(best_fused)[::-1][1]
        narrow1 = self._compute_fbcca_narrow(data_lat, top1)
        narrow2 = self._compute_fbcca_narrow(data_lat, top2_idx)
        narrow_ratio = narrow1 / (narrow2 + 1e-10)
        if narrow_ratio > 1.3:
            return top1
        elif narrow_ratio < 0.77:
            return top2_idx

        # ── Phase 10: Final FBCCA fallback ──
        if fb_margin > 1.08:
            return fb_pred

        # ── Phase 11: Narrow-band CCA tiebreaker ──
        nb_cca_scores = []
        for ti in range(len(self.freqs)):
            freq = self.freqs[ti]
            nh = self._select_n_harmonics(freq, data_lat.shape[1] / self.srate)
            sample = np.linspace(0, (data_lat.shape[1] - 1) / self.srate,
                                 data_lat.shape[1], endpoint=True)
            rows = []
            for h in range(1, nh + 1):
                phase = 2 * np.pi * freq * h * sample
                rows.append(np.sin(phase))
                rows.append(np.cos(phase))
            Y = np.vstack(rows).T
            nyq = self.srate / 2
            low = max(2, freq - 0.5) / nyq
            high = min(nyq - 1, freq + 0.5) / nyq
            b_bp, a_bp = signal.butter(4, [low, high], btype='bandpass')
            filtered = signal.filtfilt(b_bp, a_bp, data_lat, axis=1)
            nb_cca_scores.append(self._cca_score(filtered.T, Y))
        nb_cca_pred = int(np.argmax(nb_cca_scores))
        if nb_cca_pred != top1 and nb_cca_scores[nb_cca_pred] > narrow1 * 1.02:
            return nb_cca_pred

        return tentative

    def _detect_short(self, data, train_data=None):
        delay = self.latency_ms
        data_lat = data[:, delay:] if 0 < delay < data.shape[1] else data
        trimmedLen = data_lat.shape[1]

        fft_scores = self._compute_fft_snr(data_lat)
        fft_scores_sup = self._adjacent_freq_suppression(fft_scores)

        trimmed_templates = self._build_adaptive_templates(
            self.freqs, trimmedLen / self.srate)
        _, fbcca_scores = self._compute_fbcca_scores(
            data_lat, trimmed_templates)

        fused = self._adaptive_weighted_fusion([fft_scores_sup, fbcca_scores])
        top2 = np.sort(fused)[::-1][:2]
        margin = top2[0] / (top2[1] + 1e-6)

        if margin > 1.3:
            return int(np.argmax(fused))

        # v2.1: Enhanced early-return for short windows.
        # In short windows (0.5-1.5s), CCA/FBCCA is inherently more
        # reliable than fusion because fusion between two noisy methods
        # can produce worse results than the better individual method.
        # We add an early return: if FBCCA margin is modestly high,
        # trust it directly rather than diluting with FFT.
        fb_top2_s = np.sort(fbcca_scores)[::-1][:2]
        fb_margin_s = fb_top2_s[0] / (fb_top2_s[1] + 1e-6)
        if fb_margin_s > 1.25:
            return int(np.argmax(fbcca_scores))

        enh_templates = self._build_adaptive_templates(
            self.freqs, trimmedLen / self.srate)
        _, fbcca_enh = self._compute_fbcca_scores(
            data_lat, enh_templates,
            self._enhanced_filter_banks, self._enhanced_fb_weights)

        fused_enh = self._adaptive_weighted_fusion([fft_scores_sup, fbcca_enh])
        top2_enh = np.sort(fused_enh)[::-1][:2]
        margin_enh = top2_enh[0] / (top2_enh[1] + 1e-6)

        if margin_enh > margin:
            return int(np.argmax(fused_enh))

        # v2.1: Enhanced FBCCA early-return for short windows.
        enh_fb_top2 = np.sort(fbcca_enh)[::-1][:2]
        enh_fb_margin = enh_fb_top2[0] / (enh_fb_top2[1] + 1e-6)
        if enh_fb_margin > 1.25 and enh_fb_margin > margin_enh:
            return int(np.argmax(fbcca_enh))

        # Narrow-band refinement for short window
        top1 = int(np.argmax(fused))
        top2_idx = np.argsort(fused)[::-1][1]
        narrow1 = self._compute_fbcca_narrow(data_lat, top1)
        narrow2 = self._compute_fbcca_narrow(data_lat, top2_idx)
        if narrow1 > narrow2 * 1.2:
            return top1
        elif narrow2 > narrow1 * 1.2:
            return top2_idx

        # v2.1: Final low-margin fallback.
        # When all margins are very low, the data is genuinely ambiguous.
        # In this case, standard FBCCA is the most reliable single method.
        # Returning its prediction avoids fusion-induced noise amplification.
        if fb_margin_s > 1.05:
            return int(np.argmax(fbcca_scores))

        return int(np.argmax(fused))

    # Legacy compatibility
    def _two_stage_reevaluate(self, data, base_pred, _base_combined=None):
        trimmedLen = data.shape[1]
        enhanced_templates = self._build_templates_for_len(
            self.freqs, 5, trimmedLen / self.srate)
        p_enh, c_enh = self._run_fbcca(
            data, enhanced_templates,
            self._enhanced_filter_banks, self._enhanced_fb_weights)
        sorted_enh = np.sort(c_enh)[::-1]
        enh_margin = sorted_enh[0] / (sorted_enh[1] + 1e-6)
        if enh_margin > 1.3 and p_enh != base_pred:
            return p_enh
        if self.trca_templates is not None:
            X_trca = self._preprocess(
                data[self.chans] if self.chans else data,
                targetLen=self.trca_targetLen)
            trca_scores = self._trca_ensemble_score(X_trca)
            trca_pred = int(np.argmax(trca_scores))
            if trca_pred == p_enh and trca_pred != base_pred:
                return trca_pred
        return None
