# src/data_operations/patient_splits.py
import re
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split

# Match "..._P_00012_..." in CBIS-DDSM paths
_PAT_RE = re.compile(r'_P_(\d{5})')

def patient_id_from_path(p: str) -> str:
    p = p.replace("\\", "/")
    m = _PAT_RE.search(p)
    return m.group(1) if m else p  # fallback: whole path

def group_by_patient(paths, labels):
    """
    Returns:
      pid_to_idx: dict[pid] -> list[int] indices in paths/labels
      pid_major_label: dict[pid] -> majority class label (for stratification)
    """
    pid_to_idx = defaultdict(list)
    for i, p in enumerate(paths):
        pid_to_idx[patient_id_from_path(p)].append(i)

    pid_major_label = {}
    for pid, idxs in pid_to_idx.items():
        # majority label per patient (binary labels expected: 0/1)
        y = [labels[i] for i in idxs]
        # tie-break goes to the positive class if counts equal
        vals, cnts = np.unique(y, return_counts=True)
        pid_major_label[pid] = int(vals[np.argmax(cnts)])

    return pid_to_idx, pid_major_label

def split_patients(pid_list, pid_labels, train=0.70, val=0.15, test=0.15, seed=42):
    """
    Splits patients (not images) into train/val/test with stratification by majority label.
    """
    assert abs((train + val + test) - 1.0) < 1e-6, "splits must sum to 1.0"
    rng = seed

    pids = np.array(pid_list)
    y_pid = np.array([pid_labels[pid] for pid in pids])

    # train vs (val+test)
    p_train, p_rest, y_train, y_rest = train_test_split(
        pids, y_pid, test_size=(1 - train), stratify=y_pid, random_state=rng, shuffle=True
    )

    # val vs test (keep original class balance)
    val_frac_of_rest = val / (val + test)
    p_val, p_test, _, _ = train_test_split(
        p_rest, y_rest, test_size=(1 - val_frac_of_rest), stratify=y_rest, random_state=rng, shuffle=True
    )

    return set(p_train), set(p_val), set(p_test)
