/**
 * Extract CLIP image embeddings for a list of files.
 *
 * Uses the same checkpoint and the same quantization the browser loads, because
 * a probe fitted on fp32 embeddings and served against q8 ones is a train/serve
 * skew you cannot see and cannot debug.
 *
 * Usage: node tools/embed.mjs <manifest.json> <out.bin>
 *   manifest.json: { "files": ["/abs/path.jpg", …] }
 *   out.bin:       float32 [n, 512], row-major, L2-normalized
 */
import { readFileSync, writeFileSync } from "node:fs";

import { AutoProcessor, CLIPVisionModelWithProjection, RawImage } from "@huggingface/transformers";

const MODEL = "Xenova/clip-vit-base-patch32";
const DTYPE = "q8";
const BATCH = 16;

const [manifestPath, outPath] = process.argv.slice(2);
if (!manifestPath || !outPath) {
  console.error("usage: node tools/embed.mjs <manifest.json> <out.bin>");
  process.exit(1);
}

const { files } = JSON.parse(readFileSync(manifestPath, "utf8"));
const processor = await AutoProcessor.from_pretrained(MODEL);
const vision = await CLIPVisionModelWithProjection.from_pretrained(MODEL, { dtype: DTYPE });

const out = new Float32Array(files.length * 512);
const started = Date.now();

for (let i = 0; i < files.length; i += BATCH) {
  const slice = files.slice(i, i + BATCH);
  const images = await Promise.all(slice.map((f) => RawImage.read(f)));
  const { image_embeds } = await vision(await processor(images));
  const rows = image_embeds.tolist();
  rows.forEach((row, j) => {
    const norm = Math.hypot(...row);
    for (let k = 0; k < 512; k++) out[(i + j) * 512 + k] = row[k] / norm;
  });
  if (i % (BATCH * 10) === 0 || i + BATCH >= files.length) {
    const done = Math.min(i + BATCH, files.length);
    const rate = done / ((Date.now() - started) / 1000);
    const left = Math.round((files.length - done) / rate);
    process.stderr.write(
      `\r${done}/${files.length}  ${rate.toFixed(1)}/s  ~${left}s left        `,
    );
  }
}

writeFileSync(outPath, Buffer.from(out.buffer));
process.stderr.write(`\ndone in ${Math.round((Date.now() - started) / 1000)}s → ${outPath}\n`);
