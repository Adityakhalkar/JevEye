"""
List the distillation images once, in one order, for teacher and student alike.

The manifest is the single source of truth for which image is which. The
embedder reads it to produce the teacher vectors and the trainer reads it to
load the inputs, so the pairing cannot drift.

Usage: python3 tools/teacher_manifest.py <images-dir>
"""
import json
import sys
from pathlib import Path

images = Path(sys.argv[1]).resolve()
files = sorted(str(p) for p in images.rglob("*.jpg"))
json.dump({"files": files}, open(images / "manifest.json", "w"))
print(f"{len(files)} images listed in {images}/manifest.json")
