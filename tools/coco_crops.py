"""
Cut COCO's labelled instances into crops a classifier can be fitted on.

Flowers gives one label per photograph; COCO labels objects inside one, so the
training examples here are crops rather than images. Three decisions matter
more than they look:

- **Context.** COCO boxes are tight. CLIP was trained on photographs, not on
  objects sheared at the edge, so each crop is padded outward.
- **Balance.** `person` appears about eleven thousand times in val2017 and
  `hair drier` eleven. Fitted unchecked, the probe learns the prior rather than
  the classes, so each class is capped.
- **Splitting by image, not by crop.** Two crops from one photograph share
  lighting, camera and often background; letting one land in train and the
  other in test measures memorisation.

Usage: python3 tools/coco_crops.py <coco-dir> <out-dir> [--cap N]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

coco_dir, out_dir = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
cap = int(sys.argv[sys.argv.index("--cap") + 1]) if "--cap" in sys.argv else 300
(out_dir / "crops").mkdir(parents=True, exist_ok=True)

SEED = 0
# Below this a crop is a handful of pixels upscaled to 224: noise with a label.
MIN_SIDE = 40
# Outward padding, as a share of the box, so the object keeps its surroundings.
PAD = 0.12
CALIB_SHARE, TEST_SHARE = 0.18, 0.22

rng = np.random.default_rng(SEED)
data = json.load(open(coco_dir / "annotations" / "instances_val2017.json"))

# COCO category ids are sparse (1..90 with gaps); the probe wants 0..79.
categories = sorted(data["categories"], key=lambda c: c["id"])
name_of = {c["id"]: c["name"] for c in categories}
index_of = {c["id"]: i for i, c in enumerate(categories)}
images = {im["id"]: im for im in data["images"]}
print(f"{len(categories)} categories, {len(images)} images, {len(data['annotations'])} annotations")

# Split by image before touching any annotation.
image_ids = np.array(sorted(images))
rng.shuffle(image_ids)
n_calib = int(len(image_ids) * CALIB_SHARE)
n_test = int(len(image_ids) * TEST_SHARE)
split_of = {}
for i, image_id in enumerate(image_ids):
    split_of[int(image_id)] = "calib" if i < n_calib else "test" if i < n_calib + n_test else "train"

by_split_class = defaultdict(list)
for ann in data["annotations"]:
    if ann.get("iscrowd"):
        continue
    x, y, w, h = ann["bbox"]
    if w < MIN_SIDE or h < MIN_SIDE:
        continue
    by_split_class[(split_of[ann["image_id"]], ann["category_id"])].append(ann)

# Cap per class per split, proportionally: a test set skewed to `person` reports
# how well the probe does on people and calls it an average.
caps = {"train": cap, "calib": max(20, cap // 4), "test": max(20, cap // 3)}
chosen = defaultdict(list)
for (split, category), anns in by_split_class.items():
    take = min(len(anns), caps[split])
    picked = rng.permutation(len(anns))[:take]
    chosen[split].extend(anns[i] for i in picked)

manifests = {}
labels = {}
index = 0
for split in ["train", "calib", "test"]:
    files, ys = [], []
    for ann in chosen[split]:
        meta = images[ann["image_id"]]
        source = coco_dir / "val2017" / meta["file_name"]
        if not source.exists():
            continue
        x, y, w, h = ann["bbox"]
        px, py = w * PAD, h * PAD
        box = (
            max(0, int(x - px)),
            max(0, int(y - py)),
            min(meta["width"], int(x + w + px)),
            min(meta["height"], int(y + h + py)),
        )
        if box[2] - box[0] < MIN_SIDE or box[3] - box[1] < MIN_SIDE:
            continue
        index += 1
        path = out_dir / "crops" / f"crop_{index:06d}.jpg"
        try:
            Image.open(source).convert("RGB").crop(box).save(path, quality=92)
        except OSError:
            continue
        files.append(str(path))
        ys.append(index_of[ann["category_id"]])
    manifests[split] = files
    labels[split] = ys
    counts = np.bincount(ys, minlength=len(categories))
    print(
        f"{split:>5}: {len(files):>6} crops, "
        f"{int((counts > 0).sum())} classes present, "
        f"largest class {int(counts.max())}"
    )

for split, files in manifests.items():
    json.dump({"files": files}, open(out_dir / f"{split}.json", "w"))
json.dump(
    {
        "train_labels": labels["train"],
        "calib_labels": labels["calib"],
        "test_labels": labels["test"],
        "classes": len(categories),
        "class_names": [c["name"] for c in categories],
        "seed": SEED,
        "cap_per_class": cap,
    },
    open(out_dir / "meta.json", "w"),
)
print(f"wrote manifests to {out_dir}")
