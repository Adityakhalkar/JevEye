/**
 * Extract CLIP text embeddings for a label set, through its prompt template.
 *
 * These are the zero-shot classifier: scoring an image against them is exactly
 * what the app does today, so they double as the baseline a probe has to beat.
 *
 * Usage: node tools/embed-text.mjs <vocabulary-id> <out.bin>
 */
import { writeFileSync } from "node:fs";

import { AutoTokenizer, CLIPTextModelWithProjection } from "@huggingface/transformers";

import { VOCABULARIES } from "../src/lib/vocab/index.ts";

const MODEL = "Xenova/clip-vit-base-patch32";

const [vocabId, outPath] = process.argv.slice(2);
const vocab = VOCABULARIES[vocabId];
if (!vocab || !outPath) {
  console.error("usage: node tools/embed-text.mjs <vocabulary-id> <out.bin>");
  process.exit(1);
}

const tokenizer = await AutoTokenizer.from_pretrained(MODEL);
const text = await CLIPTextModelWithProjection.from_pretrained(MODEL, { dtype: "q8" });

const prompts = vocab.labels.map((l) => vocab.hypothesis.replace("{}", l));
const { text_embeds } = await text(await tokenizer(prompts, { padding: true, truncation: true }));
const rows = text_embeds.tolist();

const out = new Float32Array(rows.length * 512);
rows.forEach((row, i) => {
  const norm = Math.hypot(...row);
  for (let k = 0; k < 512; k++) out[i * 512 + k] = row[k] / norm;
});

writeFileSync(outPath, Buffer.from(out.buffer));
console.error(`${rows.length} label embeddings → ${outPath}`);
