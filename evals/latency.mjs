/**
 * Where a question's time actually goes.
 *
 * Every stage here has been measured at some point and never in one place, so
 * the budget has been a matter of belief. Each is timed directly, at the
 * settings the app ships, and reported as a share of the whole — because the
 * useful question is not "is it fast" but "which stage would repay work".
 *
 * Node on CPU. A browser on WebGPU will be faster for the neural stages and no
 * faster for the network one, which moves the balance rather than the totals.
 */
import { RawImage, pipeline } from "@huggingface/transformers";

import { DETECT_EDGE_LIVE, DETECT_EDGE_STILL } from "../src/lib/vision.ts";

const IMAGE = process.env.LATENCY_IMAGE ?? "/tmp/dog.jpg";
const REPEATS = 3;

const median = (xs) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];

async function time(label, fn, repeats = REPEATS) {
  await fn(); // warm up: graph optimisation is not latency
  const runs = [];
  for (let i = 0; i < repeats; i++) {
    const t = Date.now();
    await fn();
    runs.push(Date.now() - t);
  }
  return { label, ms: median(runs) };
}

export async function evaluateLatency() {
  const { AutoProcessor, AutoTokenizer, CLIPTextModelWithProjection, CLIPVisionModelWithProjection } =
    await import("@huggingface/transformers");
  const CLIP = "Xenova/clip-vit-base-patch32";
  const image = await RawImage.read(IMAGE);

  const processor = await AutoProcessor.from_pretrained(CLIP);
  const vision = await CLIPVisionModelWithProjection.from_pretrained(CLIP, { dtype: "q8" });
  const tokenizer = await AutoTokenizer.from_pretrained(CLIP);
  const text = await CLIPTextModelWithProjection.from_pretrained(CLIP, { dtype: "q8" });

  const tiles = [];
  const w = Math.floor(image.width / 4);
  const h = Math.floor(image.height / 4);
  for (let r = 0; r < 4; r++)
    for (let c = 0; c < 4; c++) tiles.push(await image.crop([c * w, r * h, c * w + w, r * h + h]));

  const detector = await pipeline("object-detection", "Xenova/detr-resnet-50", { dtype: "q8" });
  const setEdge = (edge) => {
    const p = detector.processor?.image_processor;
    if (p) p.size = { shortest_edge: edge, longest_edge: Math.round(edge * 1.67) };
  };

  const stages = [];
  stages.push(await time("embed one frame (the live gate)", async () => {
    await vision(await processor([image]));
  }));
  stages.push(await time("embed the 4×4 coverage grid", async () => {
    await vision(await processor(tiles));
  }));
  stages.push(await time("embed 102 label prompts (cached after first)", async () => {
    await text(tokenizer(Array.from({ length: 102 }, (_, i) => `a photo of a flower ${i}`), {
      padding: true,
      truncation: true,
    }));
  }, 1));
  setEdge(DETECT_EDGE_LIVE);
  stages.push(await time(`detect at ${DETECT_EDGE_LIVE}px (live)`, async () => {
    await detector(image, { threshold: 0.7 });
  }));
  setEdge(DETECT_EDGE_STILL);
  stages.push(await time(`detect at ${DETECT_EDGE_STILL}px (stills)`, async () => {
    await detector(image, { threshold: 0.5 });
  }));

  console.log("\nWHERE THE TIME GOES (Node, CPU, median of 3)");
  const widest = Math.max(...stages.map((s) => s.label.length));
  for (const s of stages) {
    console.log(`  ${s.label.padEnd(widest)}  ${String(s.ms).padStart(6)} ms`);
  }

  const embed = stages[0].ms;
  const grid = stages[1].ms;
  const live = stages[3].ms;
  const still = stages[4].ms;
  const JEV = 1124; // measured separately: median round trip, three questions

  console.log("\n  a still question, identify over objects:");
  console.log(`    ${String(grid).padStart(6)} ms  coverage grid`);
  console.log(`    ${String(still).padStart(6)} ms  detector`);
  console.log(`    ${String(JEV * 2).padStart(6)} ms  Jev, twice (read the question, judge the facts)`);
  console.log(`    ${String(grid + still + JEV * 2).padStart(6)} ms  total, of which ${((JEV * 2) / (grid + still + JEV * 2) * 100).toFixed(0)}% is waiting on the network`);

  console.log("\n  one live sample, detector every second frame:");
  console.log(`    ${String(embed).padStart(6)} ms  embed the frame`);
  console.log(`    ${String(Math.round(live / 2)).padStart(6)} ms  detector, amortised`);
  console.log(`    ${String(embed + Math.round(live / 2)).padStart(6)} ms  total against a 300 ms budget`);
  return 0;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  process.exit(await evaluateLatency());
}
