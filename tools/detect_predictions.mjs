/**
 * Run the detector over COCO val images and write predictions for scoring.
 *
 * The detector has been trusted throughout this project on the strength of one
 * photograph of a dog. mAP is the number that says whether that trust
 * generalises, and it is the one number a detection model is normally judged by.
 *
 * Runs at the same input resolution the app uses, because a score measured at a
 * resolution nothing ships is a score for a different model.
 *
 * Usage: node tools/detect_predictions.mjs <coco-dir> <out.json> [--limit N] [--edge N]
 */
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import { RawImage, pipeline } from "@huggingface/transformers";

const [cocoDir, outPath] = process.argv.slice(2);
const limit = process.argv.includes("--limit")
  ? Number(process.argv[process.argv.indexOf("--limit") + 1])
  : 500;
const edge = process.argv.includes("--edge")
  ? Number(process.argv[process.argv.indexOf("--edge") + 1])
  : 560;
// Low, because mAP integrates over the whole precision-recall curve: cutting
// early throws away the recall the metric is asking about.
const THRESHOLD = 0.05;

const annotations = JSON.parse(
  readFileSync(join(cocoDir, "annotations", "instances_val2017.json"), "utf8"),
);
const categoryId = new Map(annotations.categories.map((c) => [c.name, c.id]));
const images = annotations.images.slice(0, limit);

const detect = await pipeline("object-detection", "Xenova/detr-resnet-50", { dtype: "q8" });
const processor = detect.processor?.image_processor;
if (processor) {
  processor.size = { shortest_edge: edge, longest_edge: Math.round(edge * 1.67) };
}

const predictions = [];
const started = Date.now();
for (const [i, meta] of images.entries()) {
  const image = await RawImage.read(join(cocoDir, "val2017", meta.file_name));
  const found = await detect(image, { threshold: THRESHOLD, percentage: false });
  for (const f of found) {
    const id = categoryId.get(f.label);
    if (id === undefined) continue;
    predictions.push({
      image_id: meta.id,
      category_id: id,
      bbox: [f.box.xmin, f.box.ymin, f.box.xmax - f.box.xmin, f.box.ymax - f.box.ymin],
      score: f.score,
    });
  }
  if ((i + 1) % 50 === 0) {
    const rate = (i + 1) / ((Date.now() - started) / 1000);
    process.stderr.write(
      `\r${i + 1}/${images.length}  ${rate.toFixed(1)} img/s  ${predictions.length} boxes   `,
    );
  }
}
writeFileSync(outPath, JSON.stringify(predictions));
process.stderr.write(
  `\ndone: ${images.length} images at edge ${edge}, ` +
    `${((Date.now() - started) / images.length).toFixed(0)}ms each -> ${outPath}\n`,
);
