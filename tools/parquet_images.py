"""
Pull plain images out of parquet shards, for distillation.

No labels are read and none are needed: the teacher supplies every target, so
any photograph is a usable training example. That is the property that makes
distillation cheap to feed — the constraint is disk and time, not annotation.

Usage: python3 tools/parquet_images.py <shard-dir> <out-dir> [--max N]
"""
import io
import sys
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

shard_dir, out_dir = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
most = int(sys.argv[sys.argv.index("--max") + 1]) if "--max" in sys.argv else 0
out_dir.mkdir(parents=True, exist_ok=True)

# Downscale on the way in: the teacher sees 224px, so storing more is disk spent
# on pixels nothing will ever read.
LONGEST = 384

written = 0
for shard in sorted(shard_dir.glob("*.parquet")):
    table = pq.read_table(shard, columns=["image"]) if "image" in pq.ParquetFile(shard).schema_arrow.names else pq.read_table(shard)
    column = "image" if "image" in table.column_names else table.column_names[0]
    for row in table.column(column).to_pylist():
        data = row["bytes"] if isinstance(row, dict) else row
        if not data:
            continue
        try:
            image = Image.open(io.BytesIO(data)).convert("RGB")
        except OSError:
            continue
        if max(image.size) > LONGEST:
            scale = LONGEST / max(image.size)
            image = image.resize((round(image.width * scale), round(image.height * scale)))
        written += 1
        image.save(out_dir / f"hf_{written:06d}.jpg", quality=90)
        if most and written >= most:
            print(f"{written} images -> {out_dir}")
            sys.exit(0)
    print(f"  {shard.name}: running total {written}")

print(f"{written} images -> {out_dir}")
