# src/data_visualisation/calibration_report.py
import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import os
os.environ["MPLBACKEND"] = "Agg" 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

def _as_prob(p):
    p = np.asarray(p, dtype=float).ravel()
    # if logits sneaked in, map to (0,1)
    if p.max() > 1.0 or p.min() < 0.0:
        p = 1.0 / (1.0 + np.exp(-p))
    return np.clip(p, 0.0, 1.0)

def brier_score(y_true, p):
    y = np.asarray(y_true).astype(float).ravel()
    p = np.asarray(p).astype(float).ravel()
    return float(np.mean((p - y) ** 2))

def ece_binary(y_true, p, n_bins=15):
    y = np.asarray(y_true).astype(int).ravel()
    p = np.asarray(p).astype(float).ravel()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx  = np.digitize(p, bins) - 1  # 0..n_bins-1
    ece = 0.0
    total = len(y)
    for b in range(n_bins):
        mask = (idx == b)
        if not np.any(mask):
            continue
        conf = p[mask].mean()
        acc  = (y[mask] == (p[mask] >= 0.5)).mean()
        w    = mask.mean()
        ece += w * abs(acc - conf)
    return float(ece)

def mce_binary(y_true, p, n_bins=15):
    y = np.asarray(y_true).astype(int).ravel()
    p = np.asarray(p).astype(float).ravel()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx  = np.digitize(p, bins) - 1
    mce = 0.0
    for b in range(n_bins):
        mask = (idx == b)
        if not np.any(mask):
            continue
        conf = p[mask].mean()
        acc  = y[mask].mean()
        mce = max(mce, abs(acc - conf))
    return float(mce)


def find_threshold_at_specificity(y_true, y_prob, target_spec=0.90):
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob).astype(float).ravel()

    fpr, tpr, thr = roc_curve(y_true, y_prob)  # positive class = 1
    spec = 1.0 - fpr
    target_spec = float(target_spec)

    # 1) First try: all thresholds that *actually* meet spec ≥ target_spec
    idx = np.where(spec >= target_spec)[0]

    # drop trivial all-negative points (tpr == 0)
    idx = idx[tpr[idx] > 0]

    if idx.size > 0:
        # among feasible, pick the one with max sensitivity
        best = idx[np.argmax(tpr[idx])]
        return float(thr[best]), float(tpr[best]), float(spec[best])

    # 2) If no non-trivial threshold meets spec ≥ target_spec,
    #    pick the point whose specificity is *closest* to the target.
    j = int(np.argmin(np.abs(spec - target_spec)))
    return float(thr[j]), float(tpr[j]), float(spec[j])

def plot_reliability_diagram(y_true, p_uncal, p_cal=None, n_bins=15, out_path=None, title_suffix=""):
    y = np.asarray(y_true).astype(int).ravel()

    def _bin_curve(p):
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        mids = 0.5 * (bins[:-1] + bins[1:])
        idx  = np.digitize(p, bins) - 1
        confs, accs, counts = [], [], []
        for b in range(n_bins):
            m = (idx == b)
            if not np.any(m):
                confs.append(np.nan); accs.append(np.nan); counts.append(0)
            else:
                confs.append(p[m].mean())
                accs.append(y[m].mean())
                counts.append(int(m.sum()))
        confs = np.array(confs, float)
        accs  = np.array(accs, float)
        return mids, confs, accs, np.array(counts)

    mids, c_u, a_u, _ = _bin_curve(np.asarray(p_uncal).ravel())
    fig = plt.figure(figsize=(5.5, 5.5))
    ax = plt.gca()
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Perfect", alpha=0.7)

    ax.plot(c_u, a_u, marker="o", linewidth=1.5, label="Uncalibrated")
    if p_cal is not None:
        _, c_c, a_c, _ = _bin_curve(np.asarray(p_cal).ravel())
        ax.plot(c_c, a_c, marker="s", linewidth=1.5, label="Calibrated (Temp)")

    ax.set_xlabel("Predicted probability (confidence)")
    ax.set_ylabel("Observed accuracy")
    ax.set_title(f"Reliability diagram{title_suffix}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=180)
        plt.close(fig)
    else:
        plt.show()

def calibration_report(y_true, p_uncal, p_cal=None, prefix="reports/calibration/run"):
    y = np.asarray(y_true, dtype=int).ravel()
    p_u = _as_prob(p_uncal)

    # core metrics
    brier_u = brier_score(y, p_u)
    ece_u   = ece_binary(y, p_u, n_bins=15)
    mce_u   = mce_binary(y, p_u, n_bins=15)
    auc_u   = roc_auc_score(y, p_u)
    ap_u    = average_precision_score(y, p_u)

    targets = [0.80, 0.90, 0.95]
    sens_rows_u = [("Uncalibrated", sp, *find_threshold_at_specificity(y, p_u, sp))
                   for sp in targets]

    print("\n=== Calibration Report (Uncalibrated) ===")
    print(f"AUC: {auc_u:.4f} | PR-AUC: {ap_u:.4f}")
    print(f"Brier: {brier_u:.4f} | ECE: {ece_u:.4f} | MCE: {mce_u:.4f}")
    for tag, sp, thr, sens, spec in sens_rows_u:
        print(f"{tag}: @Spec≥{sp:.2f}  thr={thr:.3f}  Sens={sens:.3f}  Spec={spec:.3f}")

    # optional calibrated block
    p_c_vec, brier_c, ece_c, mce_c, auc_c, ap_c, sens_rows_c = (None,)*7
    if p_cal is not None:
        p_c = _as_prob(p_cal)
        brier_c = brier_score(y, p_c)
        ece_c   = ece_binary(y, p_c, n_bins=15)
        mce_c   = mce_binary(y, p_c, n_bins=15)
        auc_c   = roc_auc_score(y, p_c)
        ap_c    = average_precision_score(y, p_c)
        sens_rows_c = [("Calibrated", sp, *find_threshold_at_specificity(y, p_c, sp))
                       for sp in targets]

        print("\n=== Calibration Report (Calibrated: Temperature) ===")
        print(f"AUC: {auc_c:.4f} | PR-AUC: {ap_c:.4f}")
        print(f"Brier: {brier_c:.4f} | ECE: {ece_c:.4f} | MCE: {mce_c:.4f}")
        for tag, sp, thr, sens, spec in sens_rows_c:
            print(f"{tag}: @Spec≥{sp:.2f}  thr={thr:.3f}  Sens={sens:.3f}  Spec={spec:.3f}")

    # plot reliability
    png = f"{prefix}_reliability.png"
    plot_reliability_diagram(y, p_u, (p_c if p_cal is not None else None),
                             n_bins=15, out_path=png, title_suffix="")
    print(f"[saved] Reliability diagram -> {png}")

    # return (and optionally log) a summary
    summary = {
        "uncal": {"auc": auc_u, "prauc": ap_u, "brier": brier_u, "ece": ece_u, "mce": mce_u,
                  "ops": {sp: {"thr": t, "sens": s, "spec": c} for (_, sp, t, s, c) in sens_rows_u}}
    }
    if p_cal is not None:
        summary["cal"] = {"auc": auc_c, "prauc": ap_c, "brier": brier_c, "ece": ece_c, "mce": mce_c,
                          "ops": {sp: {"thr": t, "sens": s, "spec": c} for (_, sp, t, s, c) in sens_rows_c}}
    return summary