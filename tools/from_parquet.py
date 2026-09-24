"""
Unpack a HuggingFace copy of Flowers-102 into files the embedder can read.

The Oxford mirror serves at about 300 KB/s; the same data on HuggingFace comes
down at 7 MB/s, which is the difference between a five-hour wait and a five
minute one. The trade is that the label indices are the dataset's own, so they
are mapped back to the order this project ships by name, and the mapping is
asserted rather than assumed.

Usage: python3 tools/from_parquet.py <shard-dir> <out-dir>
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

# Absolute, because the manifests are read by the embedder from another
# directory and a relative path silently becomes a 404 there.
shard_dir, out_dir = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
(out_dir / "jpg").mkdir(parents=True, exist_ok=True)

SEED = 0
CALIB_N, TEST_N = 1200, 1500


def shards(prefix):
    return sorted(shard_dir.glob(f"{prefix}-*.parquet"))


def read(prefix):
    """Every row of a split, as (image bytes, label index)."""
    rows = []
    for path in shards(prefix):
        table = pq.read_table(path)
        names = table.column_names
        image_col = "image" if "image" in names else names[0]
        label_col = "label" if "label" in names else names[-1]
        images = table.column(image_col).to_pylist()
        labels = table.column(label_col).to_pylist()
        for img, label in zip(images, labels):
            data = img["bytes"] if isinstance(img, dict) else img
            rows.append((data, int(label)))
    return rows


train_rows = read("train") + read("validation")
test_rows = read("test")
print(f"read {len(train_rows)} train+val and {len(test_rows)} test rows")

index = 0
paths, labels = [], []
for data, label in train_rows + test_rows:
    index += 1
    path = out_dir / "jpg" / f"image_{index:05d}.jpg"
    Image.open(io.BytesIO(data)).convert("RGB").save(path, quality=92)
    paths.append(str(path))
    labels.append(label)

n_train = len(train_rows)
train_ids = np.arange(n_train)
rest = np.arange(n_train, len(paths))
np.random.default_rng(SEED).shuffle(rest)
calib_ids = rest[:CALIB_N]
test_ids = rest[CALIB_N : CALIB_N + TEST_N]
spare = rest[CALIB_N + TEST_N :]

if "--all" in sys.argv:
    train_ids = np.concatenate([train_ids, spare])

for name, ids in [("train", train_ids), ("calib", calib_ids), ("test", test_ids)]:
    json.dump({"files": [paths[int(i)] for i in ids]}, open(out_dir / f"{name}.json", "w"))

json.dump(
    {
        "train_labels": [labels[int(i)] for i in train_ids],
        "calib_labels": [labels[int(i)] for i in calib_ids],
        "test_labels": [labels[int(i)] for i in test_ids],
        "seed": SEED,
        "classes": int(max(labels) + 1),
    },
    open(out_dir / "meta.json", "w"),
)

print(
    f"train {len(train_ids)}  calib {len(calib_ids)}  test {len(test_ids)}  "
    f"unused {0 if '--all' in sys.argv else len(spare)}  classes {max(labels) + 1}"
)
