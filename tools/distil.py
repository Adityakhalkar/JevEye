"""
Distil CLIP's image tower into something small enough to run per frame.

The student is trained to land in the *same* embedding space as the teacher,
not merely to classify well. That constraint is what makes the result useful
here: if the student's vectors sit where CLIP's do, every probe, every cached
text embedding and every downstream threshold keeps working unchanged. A
student that merely classifies would take the whole pipeline with it.

Loss is cosine distance to the teacher's unit embedding. No labels are needed,
which is the point of distillation — any photograph is a training example,
because the teacher supplies the target.

Honest expectation, written before running it: Apple's MobileCLIP was distilled
on billions of pairs and ships an 11 MB vision tower. This runs on whatever
images fit in a session. The measurement is the deliverable, not the hope.

Usage: python3 tools/distil.py <images-dir> <out-dir> [--epochs N] [--limit N]
"""
import json
import sys
import time
from pathlib import Path

# The backbone's ImageNet weights are fetched from download.pytorch.org on
# first use. That download timing out kills a run before epoch one, so cache it
# before training rather than discovering it twenty minutes in:
#   curl -sL https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth \
#     -o ~/.cache/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth
#
# ONNX export additionally needs `onnxscript`; it runs after the checkpoint is
# written, so a missing dependency costs the export but never the training.

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

images_dir, out_dir = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
epochs = int(sys.argv[sys.argv.index("--epochs") + 1]) if "--epochs" in sys.argv else 8
limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
out_dir.mkdir(parents=True, exist_ok=True)

DIM = 512
SIZE = 224
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(0)

# CLIP's own preprocessing, so the student sees what the teacher saw.
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

# The image order comes from the manifest the teacher was embedded through,
# never from a second glob. Two independent listings that disagree by one file
# would pair every student input with the wrong target, and the only symptom
# would be a loss that refuses to fall.
manifest = json.load(open(images_dir / "manifest.json"))
files = [Path(f) for f in manifest["files"]]
targets = np.fromfile(out_dir / "teacher.bin", dtype=np.float32).reshape(-1, DIM)
assert len(files) == len(targets), f"{len(files)} images against {len(targets)} teacher vectors"
if limit:
    files, targets = files[:limit], targets[:limit]

# A held-out slice, so "it learned" is measured on images it never saw.
rng = np.random.default_rng(0)
order = rng.permutation(len(files))
cut = int(len(files) * 0.9)
splits = {"train": order[:cut], "val": order[cut:]}
print(f"{len(files)} images: {len(splits['train'])} train, {len(splits['val'])} held out")


class Pairs(Dataset):
    def __init__(self, indices, train):
        self.indices = indices
        self.tf = transforms.Compose(
            ([transforms.RandomResizedCrop(SIZE, scale=(0.7, 1.0)), transforms.RandomHorizontalFlip()]
             if train
             else [transforms.Resize(SIZE), transforms.CenterCrop(SIZE)])
            + [transforms.ToTensor(), transforms.Normalize(CLIP_MEAN, CLIP_STD)]
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = int(self.indices[i])
        image = Image.open(files[j]).convert("RGB")
        return self.tf(image), torch.from_numpy(targets[j].copy())


class Student(nn.Module):
    """A small convolutional backbone with a projection into CLIP's space."""

    def __init__(self):
        super().__init__()
        backbone = mobilenet_v3_small(weights="DEFAULT")
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Sequential(
            nn.Linear(576, 1024), nn.GELU(), nn.Linear(1024, DIM)
        )

    def forward(self, x):
        x = self.pool(self.features(x)).flatten(1)
        return nn.functional.normalize(self.project(x), dim=-1)


def agreement(model, loader):
    """Mean cosine between student and teacher — 1.0 would be a perfect copy."""
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            p = model(x.to(device))
            t = nn.functional.normalize(y.to(device), dim=-1)
            total += (p * t).sum(-1).sum().item()
            n += len(x)
    return total / max(n, 1)


student = Student().to(device)
params = sum(p.numel() for p in student.parameters())
print(f"student: {params / 1e6:.1f}M parameters against the teacher's 87.8M")

# Loading in this process by default: forked workers die on macOS under Python
# 3.14, and a crash halfway through an epoch costs more than the throughput they
# buy. --workers N turns them back on where the platform tolerates it.
workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 0
train_loader = DataLoader(
    Pairs(splits["train"], True), batch_size=64, shuffle=True, num_workers=workers
)
val_loader = DataLoader(Pairs(splits["val"], False), batch_size=64, num_workers=workers)
optimiser = torch.optim.AdamW(student.parameters(), lr=3e-4, weight_decay=1e-4)
schedule = torch.optim.lr_scheduler.OneCycleLR(
    optimiser, max_lr=1e-3, total_steps=epochs * len(train_loader)
)

print(f"{'epoch':>5}  {'loss':>7}  {'agreement':>9}  {'time':>6}")
for epoch in range(1, epochs + 1):
    student.train()
    started, running = time.time(), 0.0
    for x, y in train_loader:
        x, y = x.to(device), nn.functional.normalize(y.to(device), dim=-1)
        loss = (1 - (student(x) * y).sum(-1)).mean()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        schedule.step()
        running += loss.item() * len(x)
    held_out = agreement(student, val_loader)
    print(
        f"{epoch:>5}  {running / len(splits['train']):>7.4f}  "
        f"{held_out:>9.4f}  {time.time() - started:>5.0f}s"
    )

torch.save(student.state_dict(), out_dir / "student.pt")
dummy = torch.randn(1, 3, SIZE, SIZE, device=device)
torch.onnx.export(
    student,
    dummy,
    out_dir / "student.onnx",
    input_names=["pixel_values"],
    output_names=["image_embeds"],
    dynamic_axes={"pixel_values": {0: "batch"}, "image_embeds": {0: "batch"}},
    opset_version=17,
)
json.dump(
    {"parameters": params, "agreement": agreement(student, val_loader), "epochs": epochs,
     "images": len(files)},
    open(out_dir / "student.json", "w"),
    indent=2,
)
print(f"\nwrote {out_dir}/student.onnx")
