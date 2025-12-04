# src/calibration/temperature.py
import json, os
import numpy as np
import tensorflow as tf

EPS = 1e-7

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _prob_to_logit(p):
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p) - np.log(1.0 - p)

def fit_temperature_binary(y_true, p_val, max_iters=500, lr=0.01):
    """
    Fit scalar temperature T > 0 by minimizing NLL on validation set.
    y_true: (N,1) or (N,) in {0,1}
    p_val:  (N,1) or (N,) predicted probabilities from the trained model
    Returns: float T
    """
    y = np.asarray(y_true).astype(np.float32).reshape(-1, 1)
    p = np.asarray(p_val).astype(np.float32).reshape(-1, 1)
    z = _prob_to_logit(p)  # pre-sigmoid "logits"

    T = tf.Variable(1.0, dtype=tf.float32)

    opt = tf.keras.optimizers.Adam(lr)
    for _ in range(max_iters):
        with tf.GradientTape() as tape:
            zT = z / tf.maximum(T, 1e-6)
            pT = tf.sigmoid(zT)
            # binary NLL
            nll = -tf.reduce_mean(y * tf.math.log(tf.clip_by_value(pT, EPS, 1-EPS)) +
                                  (1 - y) * tf.math.log(tf.clip_by_value(1 - pT, EPS, 1-EPS)))
        g = tape.gradient(nll, [T])
        opt.apply_gradients(zip(g, [T]))
    return float(tf.maximum(T, 1e-6).numpy())

def apply_temperature_binary(p, T):
    z = _prob_to_logit(np.asarray(p, dtype=np.float32))
    pT = _sigmoid(z / max(T, 1e-6))
    return pT

def save_temperature(path_no_ext, T, suffix="_temperature.json"):
    out = f"{path_no_ext}{suffix}"
    with open(out, "w") as f:
        json.dump({"temperature": float(T)}, f)
    return out

def load_temperature(path_no_ext, suffix="_temperature.json"):
    fpath = f"{path_no_ext}{suffix}"
    if not os.path.exists(fpath):
        return None
    with open(fpath, "r") as f:
        obj = json.load(f)
    return float(obj.get("temperature", 1.0))
