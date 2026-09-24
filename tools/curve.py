"""
Does the probe want more data, or has it stopped learning?

"Train on more" is easy to assert and easy to waste weeks on. A learning curve
answers it directly: fit the same probe on growing fractions of the training
set, score every one on the same held-out images, and look at whether the line
is still climbing. A flat tail means the ceiling is the backbone, not the
dataset, and effort belongs elsewhere.

Subsets are drawn stratified by class, because a random slice of a 102-class
set can miss classes entirely and measure that instead.

Usage: python3 tools/curve.py <workdir>
"""
import json
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
rng = np.random.default_rng(0)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def ece(probs, labels, bins=15):
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            total += (m.sum() / len(conf)) * abs(correct[m].mean() - conf[m].mean())
    return total


def fit_temperature(logits, labels):
    def nll(t):
        p = softmax(logits / max(t, 1e-3))
        return -np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, None)).mean()

    return float(minimize_scalar(nll, bounds=(0.01, 100.0), method="bounded").x)


def stratified(fraction):
    """A slice that keeps every class represented, at least once."""
    keep = []
    for cls in np.unique(train_y):
        idx = np.where(train_y == cls)[0]
        take = max(1, int(round(len(idx) * fraction)))
        keep.extend(rng.permutation(idx)[:take])
    return np.array(keep)


print(f"held-out test images: {len(test_y)}, calibration: {len(calib_y)}")
print(f"{'train':>7}  {'per class':>9}  {'accuracy':>9}  {'ECE':>6}")

rows = []
for fraction in [0.1, 0.25, 0.5, 0.75, 1.0]:
    idx = stratified(fraction)
    probe = LogisticRegression(max_iter=2000, C=10.0)
    probe.fit(train_x[idx], train_y[idx])
    W = probe.coef_.astype(np.float32)
    b = probe.intercept_.astype(np.float32)

    logits = lambda x: x @ W.T + b
    t = fit_temperature(logits(calib_x), calib_y)
    probs = softmax(logits(test_x) / t)
    accuracy = float((probs.argmax(1) == test_y).mean())
    calibration = float(ece(probs, test_y))
    per_class = len(idx) / len(np.unique(train_y))
    rows.append((len(idx), per_class, accuracy, calibration))
    print(f"{len(idx):>7}  {per_class:>9.1f}  {accuracy:>8.1%}  {calibration:>6.3f}")

# The question is whether the last step still bought anything worth having.
gain = rows[-1][2] - rows[-2][2]
print(
    f"\nthe last 25% of the data moved accuracy by {gain:+.1%}"
    f" — {'still climbing' if gain > 0.01 else 'flat: the ceiling is the backbone, not the dataset'}"
)
