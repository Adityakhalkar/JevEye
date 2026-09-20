"""
Fit a linear probe and a calibration temperature on frozen CLIP embeddings.

Two separate jobs, deliberately kept apart:

  accuracy    a multinomial logistic regression over the 512-d embeddings,
              which is the standard way to specialize CLIP without touching it
  honesty     a single temperature fitted on a held-out split so the reported
              probability means what it says

Both are measured against zero-shot on the same test images, because a number
without a baseline is not evidence.

Usage: python3 tools/fit.py <workdir> [outdir]
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression

work = sys.argv[1]
E = lambda name: np.fromfile(f"{work}/{name}.bin", dtype=np.float32).reshape(-1, 512)
meta = json.load(open(f"{work}/meta.json"))

train_x, calib_x, test_x = E("train"), E("calib"), E("test")
train_y = np.array(meta["train_labels"])
calib_y = np.array(meta["calib_labels"])
test_y = np.array(meta["test_labels"])
text = E("text")  # 102 x 512, the zero-shot classifier

CLIP_LOGIT_SCALE = 100.0


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def ece(probs, labels, bins=15):
    """Expected calibration error: mean gap between confidence and accuracy."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        total += (m.sum() / len(conf)) * abs(correct[m].mean() - conf[m].mean())
    return total


def fit_temperature(logits, labels):
    """One scalar, chosen to minimize negative log-likelihood on held-out data."""

    def nll(t):
        p = softmax(logits / max(t, 1e-3))
        return -np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, None)).mean()

    return float(minimize_scalar(nll, bounds=(0.01, 100.0), method="bounded").x)


report = {}

# ---- baseline: zero-shot, exactly what the app does today
zs_logits = lambda x: (x @ text.T) * CLIP_LOGIT_SCALE
zs_test = zs_logits(test_x)
report["zero_shot"] = {
    "accuracy": float((zs_test.argmax(1) == test_y).mean()),
    "ece_raw": float(ece(softmax(zs_test), test_y)),
}
zs_t = fit_temperature(zs_logits(calib_x), calib_y)
report["zero_shot"]["temperature"] = zs_t
report["zero_shot"]["ece_calibrated"] = float(ece(softmax(zs_test / zs_t), test_y))

# ---- linear probe on the frozen embeddings
probe = LogisticRegression(max_iter=2000, C=10.0, n_jobs=-1)
probe.fit(train_x, train_y)
W = probe.coef_.astype(np.float32)        # 102 x 512
b = probe.intercept_.astype(np.float32)   # 102

pr_logits = lambda x: x @ W.T + b
pr_test = pr_logits(test_x)
report["probe"] = {
    "accuracy": float((pr_test.argmax(1) == test_y).mean()),
    "ece_raw": float(ece(softmax(pr_test), test_y)),
    "train_images": int(len(train_y)),
}
pr_t = fit_temperature(pr_logits(calib_x), calib_y)
report["probe"]["temperature"] = pr_t
report["probe"]["ece_calibrated"] = float(ece(softmax(pr_test / pr_t), test_y))

report["test_images"] = int(len(test_y))
report["calibration_images"] = int(len(calib_y))

# ---- ship it: weights, bias and the fitted temperature the browser will load
out = sys.argv[2] if len(sys.argv) > 2 else f"{work}/out"
os.makedirs(out, exist_ok=True)
W.tofile(f"{out}/flowers.w.bin")   # [classes, 512] row-major float32
b.tofile(f"{out}/flowers.b.bin")   # [classes] float32
json.dump(
    {
        "classes": int(W.shape[0]),
        "dim": int(W.shape[1]),
        "temperature": report["probe"]["temperature"],
        "accuracy": report["probe"]["accuracy"],
        "ece": report["probe"]["ece_calibrated"],
        "trained_on": "Oxford Flowers-102 train+val, CLIP ViT-B/32 q8 embeddings",
        "zero_shot_baseline": report["zero_shot"]["accuracy"],
    },
    open(f"{out}/flowers.json", "w"),
    indent=2,
)
json.dump(report, open(f"{work}/report.json", "w"), indent=2)
print(json.dumps(report, indent=2))
