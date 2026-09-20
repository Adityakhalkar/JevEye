"""
Build the manifests and splits for fitting.

Uses the dataset's own train/val ids to fit the probe, and carves the official
test set into two disjoint halves: one to fit the temperature, one to report on.
The temperature never sees the images it is judged on, or the honesty number
would be self-graded.

Usage: python3 tools/prepare.py <workdir>
"""
import json
import sys

import numpy as np
from scipy.io import loadmat

work = sys.argv[1]
SEED = 0
CALIB_N, TEST_N = 1200, 1500

labels = loadmat(f"{work}/imagelabels.mat")["labels"][0]
sets = loadmat(f"{work}/setid.mat")

path = lambda i: f"{work}/jpg/image_{i:05d}.jpg"
# Dataset labels are 1..102; the probe works in 0..101.
label_of = lambda i: int(labels[i - 1]) - 1

train_ids = np.concatenate([sets["trnid"][0], sets["valid"][0]])
rest = sets["tstid"][0].copy()
np.random.default_rng(SEED).shuffle(rest)
calib_ids, test_ids = rest[:CALIB_N], rest[CALIB_N : CALIB_N + TEST_N]

for name, ids in [("train", train_ids), ("calib", calib_ids), ("test", test_ids)]:
    json.dump({"files": [path(int(i)) for i in ids]}, open(f"{work}/{name}.json", "w"))

json.dump(
    {
        "train_labels": [label_of(int(i)) for i in train_ids],
        "calib_labels": [label_of(int(i)) for i in calib_ids],
        "test_labels": [label_of(int(i)) for i in test_ids],
        "seed": SEED,
    },
    open(f"{work}/meta.json", "w"),
)

print(f"train {len(train_ids)}  calib {len(calib_ids)}  test {len(test_ids)}")
print(f"total images to embed: {len(train_ids) + len(calib_ids) + len(test_ids)}")
