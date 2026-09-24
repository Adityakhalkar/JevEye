"""
Is the student limited by data, or by its own size?

One agreement number cannot tell those apart, and they call for opposite
spending: more images against a bigger student. So train the same student on
growing fractions and watch the slope. Still climbing at the end means data is
the constraint and more images are worth buying. Flat means capacity is the
constraint and more images are wasted money.

The held-out tenth is fixed once, before any fraction is drawn, so every run is
scored on identical images.

Usage: python3 tools/distil_curve.py <images-dir> <out-dir> [--epochs N]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

images_dir, out_dir = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
epochs = int(sys.argv[sys.argv.index("--epochs") + 1]) if "--epochs" in sys.argv else 6
FRACTIONS = [0.125, 0.25, 0.5, 1.0]

DIM, SIZE = 512, 224
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

files = [Path(f) for f in json.load(open(images_dir / "manifest.json"))["files"]]
targets = np.fromfile(out_dir / "teacher.bin", dtype=np.float32).reshape(-1, DIM)
targets = targets / np.linalg.norm(targets, axis=1, keepdims=True)
assert len(files) == len(targets)

rng = np.random.default_rng(0)
order = rng.permutation(len(files))
cut = int(len(files) * 0.9)
pool, held_out = order[:cut], order[cut:]


class Pairs(Dataset):
    def __init__(self, indices, train):
        self.indices = indices
        self.tf = transforms.Compose(
            ([transforms.RandomResizedCrop(SIZE, scale=(0.7, 1.0)), transforms.RandomHorizontalFlip()]
             if train else [transforms.Resize(SIZE), transforms.CenterCrop(SIZE)])
            + [transforms.ToTensor(), transforms.Normalize(CLIP_MEAN, CLIP_STD)]
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = int(self.indices[i])
        return self.tf(Image.open(files[j]).convert("RGB")), torch.from_numpy(targets[j].copy())


class Student(nn.Module):
    def __init__(self):
        super().__init__()
        b = mobilenet_v3_small(weights="DEFAULT")
        self.features = b.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Sequential(nn.Linear(576, 1024), nn.GELU(), nn.Linear(1024, DIM))

    def forward(self, x):
        return nn.functional.normalize(self.project(self.pool(self.features(x)).flatten(1)), dim=-1)


def agreement(model, loader):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            total += (model(x.to(device)) * y.to(device)).sum(-1).sum().item()
            n += len(x)
    return total / max(n, 1)


val_loader = DataLoader(Pairs(held_out, False), batch_size=64)
print(f"held-out images fixed at {len(held_out)}; {epochs} epochs per point", flush=True)
print(f"{'images':>7}  {'agreement':>9}  {'minutes':>7}", flush=True)

rows = []
for fraction in FRACTIONS:
    subset = pool[: max(64, int(len(pool) * fraction))]
    loader = DataLoader(Pairs(subset, True), batch_size=64, shuffle=True)
    student = Student().to(device)
    optimiser = torch.optim.AdamW(student.parameters(), lr=3e-4, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=1e-3, total_steps=epochs * len(loader)
    )
    started = time.time()
    for _ in range(epochs):
        student.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = (1 - (student(x) * y).sum(-1)).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            schedule.step()
    score = agreement(student, val_loader)
    rows.append((len(subset), score))
    print(f"{len(subset):>7}  {score:>9.4f}  {(time.time() - started) / 60:>7.1f}", flush=True)

json.dump(
    {"epochs": epochs, "held_out": len(held_out), "points": rows},
    open(out_dir / "curve.json", "w"),
    indent=2,
)
# Reading the last step alone is the trap: a curve can look like it is still
# climbing while heading somewhere useless. The gains decay geometrically, so
# sum the rest of the series and report where the line actually ends up.
gains = [rows[i + 1][1] - rows[i][1] for i in range(len(rows) - 1)]
ratios = [gains[i + 1] / gains[i] for i in range(len(gains) - 1) if gains[i] > 0]
ratio = sum(ratios) / len(ratios) if ratios else 0.0
ceiling = rows[-1][1] + (gains[-1] * ratio / (1 - ratio) if 0 < ratio < 1 else 0.0)

print(f"\nthe last doubling moved agreement by {gains[-1]:+.4f}", flush=True)
print(f"each doubling buys {ratio:.2f} of the one before it", flush=True)
print(f"projected ceiling with unlimited data at this size: {ceiling:.3f}", flush=True)
print(
    "  -> "
    + (
        "the dataset is the limit: more images are worth buying"
        if ceiling > 0.95
        else f"the student is the limit: no amount of data passes ~{ceiling:.2f}, "
        "so a bigger or better architecture is the only lever"
    ),
    flush=True,
)
