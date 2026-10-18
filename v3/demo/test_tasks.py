"""
Test script: Task1 (datalen=3) with FBCCA, Task2 (datalen=1) with Ensemble (eCCA + eTRCA)
"""
import numpy as np
import pickle
import csv
from ss1 import ssvepDetect
from scipy import signal
from scipy.linalg import eigh
from sklearn.cross_decomposition import CCA

SRATE = 250
FREQS = [8, 9, 10, 11, 12, 13, 14, 15]
N_CLASSES = 8
N_CHANS = 6
FILTERBANK_CUTS = [(6, 70), (14, 70), (22, 70), (30, 70), (38, 70)]
N_FILTERBANKS = 5
FB_LAMBDA = 0.2
CST_N_SHIFTS = 2
CST_SHIFT_RANGE = (15, 50)
FUSION_W_ECCA = 0.85
FUSION_W_ETRCA = 0.15

DATA_DIR = r"C:\Users\Administrator\Desktop\v3\ExampleData"
MODEL_DIR = r"C:\Users\Administrator\Desktop\v3\demo"


def load_trials_by_taskid(csv_path, target_len, start_offset=0):
    with open(csv_path, "r") as f:
        lines = f.readlines()

    n_cols = len(lines[0].strip().split(","))
    raw = []
    for line in lines[1:]:
        parts = line.strip().split(",")
        if len(parts) < n_cols:
            continue
        raw.append([float(v) for v in parts])
    raw = np.array(raw, dtype=np.float64)

    task_ids = raw[:, 6]
    change_points = np.where(np.diff(task_ids) != 0)[0] + 1
    segment_starts = np.concatenate([[0], change_points])
    segment_ends = np.concatenate([change_points, [len(raw)]])

    trials = []
    for seg_start, seg_end in zip(segment_starts, segment_ends):
        seg_len = seg_end - seg_start
        if seg_len < start_offset + target_len:
            continue
        win_start = seg_start + start_offset
        win_end = win_start + target_len
        eeg = raw[win_start:win_end, :6].T.copy()
        stim_ids_in_seg = raw[seg_start:seg_end, 7].astype(int)
        stim_id = int(np.bincount(stim_ids_in_seg).argmax())
        trials.append((eeg, stim_id))
    return trials


def preprocess_trial(trial, srate=SRATE):
    data = trial.copy()
    b_notch, a_notch = signal.iirnotch(50, 30, fs=srate)
    data = signal.filtfilt(b_notch, a_notch, data, axis=1)
    nyq = srate / 2.0
    b_bp, a_bp = signal.butter(4, [6.0 / nyq, 40.0 / nyq], btype="bandpass")
    data = signal.filtfilt(b_bp, a_bp, data, axis=1)
    data = signal.detrend(data, axis=1)
    data = data - np.mean(data, axis=0, keepdims=True)
    ch_mean = np.mean(data, axis=1, keepdims=True)
    ch_std = np.std(data, axis=1, keepdims=True)
    data = (data - ch_mean) / (ch_std + 1e-6)
    return data


def predict_fb_ecca(trial, ecca_refs):
    n_comp = 2
    all_scores = np.zeros((N_FILTERBANKS, N_CLASSES))
    for band_idx, (b_band, a_band) in enumerate(ecca_refs["filterbanks"]):
        filtered_trial = signal.filtfilt(b_band, a_band, trial, axis=1)
        for cls_id in range(N_CLASSES):
            Y_ref = ecca_refs["refs_by_freq"][cls_id]
            Y_filt = signal.filtfilt(b_band, a_band, Y_ref, axis=1)
            X_t = filtered_trial.T
            Y_t = Y_filt.T
            actual_comp = min(n_comp, X_t.shape[1], Y_t.shape[1])
            if actual_comp < 1:
                continue
            cca = CCA(n_components=actual_comp)
            try:
                cca.fit(X_t, Y_t)
                X_c, Y_c = cca.transform(X_t, Y_t)
                r = np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1]
                all_scores[band_idx, cls_id] = abs(r)
            except Exception:
                pass
        band_mean = all_scores[band_idx].mean()
        band_std = all_scores[band_idx].std()
        all_scores[band_idx] = (all_scores[band_idx] - band_mean) / (band_std + 1e-6)
    scores = np.dot(ecca_refs["fb_weights"], all_scores)
    return scores


def predict_ensemble(trial, ecca_refs, etrca_model):
    score_ecca = predict_fb_ecca(trial, ecca_refs)
    # FB-eTRCA
    all_scores = np.zeros((N_FILTERBANKS, N_CLASSES))
    for band_idx, band_info in enumerate(etrca_model["bands"]):
        b_band, a_band = etrca_model["filterbanks"][band_idx]
        W_ensemble = band_info["W_ensemble"]
        templates = band_info["templates"]
        filtered = signal.filtfilt(b_band, a_band, trial, axis=1)
        y = W_ensemble.T @ filtered
        for cls_id in range(N_CLASSES):
            template = templates[cls_id]
            r = np.corrcoef(y.flatten(), template.flatten())[0, 1]
            all_scores[band_idx, cls_id] = np.sign(r) * (r ** 2)
        band_mean = all_scores[band_idx].mean()
        band_std = all_scores[band_idx].std()
        all_scores[band_idx] = (all_scores[band_idx] - band_mean) / (band_std + 1e-6)
    score_etrca = np.dot(etrca_model["fb_weights"], all_scores)

    def norm_scores(s):
        s_min, s_max = s.min(), s.max()
        if s_max - s_min > 1e-10:
            return (s - s_min) / (s_max - s_min)
        return np.zeros_like(s)

    return FUSION_W_ECCA * norm_scores(score_ecca) + FUSION_W_ETRCA * norm_scores(score_etrca)


def evaluate_fbcca(trials_raw, sd):
    correct = 0
    total = 0
    cm = np.zeros((8, 8), dtype=int)
    for eeg, true_label in trials_raw:
        res = sd.detect(eeg)
        if res == true_label:
            correct += 1
        cm[true_label, res] += 1
        total += 1
    return correct / total, cm


def evaluate_ensemble(trials_pp, ecca_refs, etrca_model):
    correct = 0
    total = 0
    cm = np.zeros((8, 8), dtype=int)
    for eeg, true_label in trials_pp:
        scores = predict_ensemble(eeg, ecca_refs, etrca_model)
        pred = int(np.argmax(scores))
        if pred == true_label:
            correct += 1
        cm[true_label, pred] += 1
        total += 1
    return correct / total, cm


def evaluate_ecca(trials_pp, ecca_refs):
    correct = 0
    total = 0
    cm = np.zeros((8, 8), dtype=int)
    for eeg, true_label in trials_pp:
        scores = predict_fb_ecca(eeg, ecca_refs)
        pred = int(np.argmax(scores))
        if pred == true_label:
            correct += 1
        cm[true_label, pred] += 1
        total += 1
    return correct / total, cm


if __name__ == "__main__":
    print("=" * 60)
    print("Task1 (datalen=3s) & Task2 (datalen=1s) 测试")
    print("=" * 60)

    # ============================================================
    # Task1: datalen=3s, S1-S6, 纯 FBCCA
    # ============================================================
    print("\n" + "=" * 60)
    print("[Task1] datalen=3s, FBCCA, S1-S6")
    print("=" * 60)

    sd_3s = ssvepDetect(SRATE, FREQS, dataLen=3)

    for subj_id in range(1, 7):
        csv_path = f"{DATA_DIR}\\Task1\\S{subj_id}.csv"
        trials = load_trials_by_taskid(csv_path, target_len=3 * SRATE, start_offset=0)
        acc, cm = evaluate_fbcca(trials, sd_3s)
        print(f"  S{subj_id}: {len(trials)} trials, acc = {acc:.4f} ({acc*100:.1f}%)")

    # ============================================================
    # Task2: datalen=1s, S7-S12, Ensemble (eCCA + eTRCA)
    # ============================================================
    print("\n" + "=" * 60)
    print("[Task2] datalen=1s, Ensemble (eCCA + eTRCA), S7-S12")
    print("=" * 60)

    with open(f"{MODEL_DIR}\\model_1s_fb_etrca.pkl", "rb") as f:
        etrca_model = pickle.load(f)
    with open(f"{MODEL_DIR}\\fb_ecca_params.pkl", "rb") as f:
        ecca_params = pickle.load(f)

    ecca_refs = {
        "refs_by_freq": ecca_params["refs_by_freq"],
        "filterbanks": ecca_params["filterbanks"],
        "fb_weights": ecca_params["fb_weights"],
        "n_harmonics": ecca_params["n_harmonics"],
        "n_components": ecca_params["n_components"],
    }

    for subj_id in range(7, 13):
        csv_path = f"{DATA_DIR}\\Task2\\S{subj_id}.csv"
        trials_raw = load_trials_by_taskid(csv_path, target_len=1 * SRATE, start_offset=0)
        trials_pp = [(preprocess_trial(eeg), lbl) for eeg, lbl in trials_raw]
        acc, cm = evaluate_ensemble(trials_pp, ecca_refs, etrca_model)
        print(f"  S{subj_id}: {len(trials_pp)} trials, acc = {acc:.4f} ({acc*100:.1f}%)")

    print("\n" + "=" * 60)
    print("测试完成。")
    print("=" * 60)
