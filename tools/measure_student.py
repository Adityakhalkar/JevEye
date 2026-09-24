"""
What the student actually learned, measured on images it never saw.

Agreement is cosine against the teacher's embedding for the same image, which
only means something next to two reference points: 1.0 is a perfect copy, and
two unrelated photographs under the teacher sit around 0.45. A number between
those is reported alongside both, because "0.82" alone reads as a pass mark
rather than as the two-thirds-of-the-way-there that it is.

Usage: python3 tools/measure_student.py   (from the workdir holding distil/)
"""
import json
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

DIM, SIZE = 512, 224
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

class Student(nn.Module):
    def __init__(self):
        super().__init__()
        b = mobilenet_v3_small(weights=None)
        self.features = b.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Sequential(nn.Linear(576, 1024), nn.GELU(), nn.Linear(1024, DIM))
    def forward(self, x):
        return nn.functional.normalize(self.project(self.pool(self.features(x)).flatten(1)), dim=-1)

student = Student().to(device)
student.load_state_dict(torch.load("distil/out/student.pt", map_location=device))
student.eval()

files = [Path(f) for f in json.load(open("distil/images/manifest.json"))["files"]]
targets = np.fromfile("distil/out/teacher.bin", dtype=np.float32).reshape(-1, DIM)
# The same held-out slice the trainer used.
order = np.random.default_rng(0).permutation(len(files))
val = order[int(len(files) * 0.9):]

tf = transforms.Compose([transforms.Resize(SIZE), transforms.CenterCrop(SIZE),
                         transforms.ToTensor(), transforms.Normalize(CLIP_MEAN, CLIP_STD)])
sims = []
with torch.no_grad():
    for i in range(0, len(val), 64):
        batch = val[i:i+64]
        x = torch.stack([tf(Image.open(files[int(j)]).convert("RGB")) for j in batch]).to(device)
        p = student(x).cpu().numpy()
        t = targets[batch]
        t = t / np.linalg.norm(t, axis=1, keepdims=True)
        sims.extend((p * t).sum(1).tolist())

sims = np.array(sims)
params = sum(p.numel() for p in student.parameters())
print(f"student {params/1e6:.1f}M params vs teacher 87.8M ({87.8/(params/1e6):.0f}x smaller)")
print(f"held-out images: {len(sims)}")
print(f"agreement with teacher: mean {sims.mean():.4f}, median {np.median(sims):.4f}")
print(f"  worst 10%: {np.percentile(sims,10):.4f}   best 10%: {np.percentile(sims,90):.4f}")
# For scale: how alike are two *different* images under the teacher?
rng = np.random.default_rng(1)
a, b = rng.choice(len(targets), 2000), rng.choice(len(targets), 2000)
base = (targets[a] / np.linalg.norm(targets[a],axis=1,keepdims=True) *
        targets[b] / np.linalg.norm(targets[b],axis=1,keepdims=True)).sum(1)
print(f"\nfor scale, two unrelated images under the teacher: mean {base.mean():.4f}")
