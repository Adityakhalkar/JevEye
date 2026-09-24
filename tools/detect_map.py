"""
Score the detector's predictions the way detectors are normally scored.

mAP over IoU 0.50:0.95 is the headline COCO number; mAP@0.50 is the looser one
people quote casually. Both are reported because they say different things: the
first is about how tightly the boxes fit, the second about whether the right
thing was found at all.

Usage: python3 tools/detect_map.py <coco-dir> <predictions.json> [--limit N]
"""
import json
import sys
from contextlib import redirect_stdout
from io import StringIO

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

coco_dir, predictions = sys.argv[1], sys.argv[2]
limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 500

with redirect_stdout(StringIO()):
    truth = COCO(f"{coco_dir}/annotations/instances_val2017.json")
    image_ids = [im["id"] for im in truth.dataset["images"][:limit]]
    detected = truth.loadRes(predictions)
    evaluation = COCOeval(truth, detected, "bbox")
    evaluation.params.imgIds = image_ids
    evaluation.evaluate()
    evaluation.accumulate()
    evaluation.summarize()

s = evaluation.stats
print(f"scored on {len(image_ids)} COCO val images")
print(f"  mAP @ IoU 0.50:0.95   {s[0]:.3f}   (the headline COCO number)")
print(f"  mAP @ IoU 0.50        {s[1]:.3f}   (was the right thing found at all)")
print(f"  mAP, large objects    {s[5]:.3f}")
print(f"  mAP, medium objects   {s[4]:.3f}")
print(f"  mAP, small objects    {s[3]:.3f}   (the live view's hard case)")
print(f"  recall @ 100 per image {s[8]:.3f}")
