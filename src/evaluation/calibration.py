# src/evaluation/calibration.py
import numpy as np
import scipy.optimize as opt
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt

def _nll_with_temp(logits, labels, T):
    # logits: (N,2) or (N,) for positive logit; here assume probs already for pos-class is p = softmax/logistic(logit/T)
    z = logits / T
    # Binary: z is logit for positive class
    p = 1.0 / (1.0 + np.exp(-z))
    eps = 1e-8
    y = labels.astype(float)
    return -np.mean(y*np.log(p+eps) + (1-y)*np.log(1-p+eps))

class TemperatureScaler:
    def __init__(self): self.T_ = 1.0
    def fit(self, val_logits, val_labels):
        obj = lambda t: _nll_with_temp(val_logits, val_labels, np.clip(t[0], 1e-3, 100.0))
        res = opt.minimize(obj, x0=[1.0], bounds=[(1e-3, 100.0)], method="L-BFGS-B")
        self.T_ = float(res.x[0])
        return self
    def predict_proba(self, logits):
        z = logits / self.T_
        p = 1.0 / (1.0 + np.exp(-z))
        return np.vstack([1-p, p]).T  # (N,2)

def expected_calibration_error(probs_pos, labels, n_bins=15):
    bins = np.linspace(0,1,n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        sel = (probs_pos >= lo) & (probs_pos < hi)
        if np.any(sel):
            conf = probs_pos[sel].mean()
            acc  = labels[sel].mean()
            ece += (sel.mean()) * abs(acc - conf)
    return float(ece)

def reliability_plot(probs_pos, labels, out_png):
    bins = np.linspace(0,1,11)
    digit = np.digitize(probs_pos, bins) - 1
    xs, ys = [], []
    for b in range(10):
        sel = digit==b
        if np.any(sel):
            xs.append(probs_pos[sel].mean())
            ys.append(labels[sel].mean())
    plt.figure(figsize=(5,5))
    plt.plot([0,1],[0,1], linestyle="--")
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed positive rate")
    plt.title("Reliability diagram")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
