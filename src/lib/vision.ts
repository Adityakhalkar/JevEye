"use client";

/**
 * The vision layer. Answers questions of fact about pixels, with a calibrated
 * confidence or an abstention. It never judges meaning: that is Jev's job, and
 * this module never calls Jev.
 *
 * Everything here runs in the browser. The image is never uploaded.
 *
 * One model, CLIP, scoring text against pixels. Instance detection was the
 * original plan, but OWL-ViT's `class_head` Cast node has no ONNX Runtime Web
 * implementation at q8, q4f16 or fp16, and only its 583 MB fp32 graph could
 * load — too much to download in a browser. So JevEye reports how much of the
 * picture each kind covers rather than how many of them there are, which is
 * both honest and more robust on dense scenes where boxes overlap anyway.
 */
import { RawImage, env, pipeline } from "@huggingface/transformers";

import {
  CALIBRATION,
  applyTemperature,
  applyTemperatureBernoulli,
  chosenReliably,
  countSummary,
  floorOrUnknown,
  poissonBinomial,
} from "./calibration";
import type { FactSheet, Plan } from "./types";
import { VOCABULARIES } from "./vocab";

env.allowLocalModels = false;

const CLIP = "Xenova/clip-vit-base-patch32";
const LONGEST_EDGE = 1280;

/**
 * ponytail: a fixed non-overlapping 4×4 grid. Coarse — a flower straddling two
 * tiles is seen twice, and one smaller than a tile's detail is diluted by its
 * background. Upgrade path when it matters: overlapping tiles at two scales,
 * or a detector once a small open-vocabulary one loads in ORT-web.
 */
const GRID_COLS = 4;
const GRID_ROWS = 4;

export type LoadProgress = { file: string; percent: number };

type Ranked = Array<{ score: number; label: string }>;
type Classifier = (
  images: unknown,
  labels: string[],
  options?: Record<string, unknown>,
) => Promise<Ranked | Ranked[]>;

let classifierPromise: Promise<Classifier> | null = null;

function device(): "webgpu" | "wasm" {
  return typeof navigator !== "undefined" && "gpu" in navigator ? "webgpu" : "wasm";
}

/** CLIP, loaded once and cached by the browser thereafter. */
export async function warmUp(onProgress?: (p: LoadProgress) => void) {
  classifierPromise ??= pipeline("zero-shot-image-classification", CLIP, {
    dtype: "q8",
    device: device(),
    progress_callback: (event: { status?: string; file?: string; progress?: number }) => {
      if (event.status === "progress" && event.file) {
        onProgress?.({ file: event.file, percent: Math.round(event.progress ?? 0) });
      }
    },
  }) as unknown as Promise<Classifier>;
  return classifierPromise;
}

/** Decode and downscale. Large photos cost time and buy nothing at CLIP's 224px. */
export async function prepare(file: Blob): Promise<RawImage> {
  const image = await RawImage.fromBlob(file);
  const longest = Math.max(image.width, image.height);
  if (longest <= LONGEST_EDGE) return image;
  const scale = LONGEST_EDGE / longest;
  return (await image.resize(
    Math.round(image.width * scale),
    Math.round(image.height * scale),
  )) as RawImage;
}

function perImage(out: Ranked | Ranked[]): Ranked[] {
  return (Array.isArray(out[0]) ? out : [out]) as Ranked[];
}

// ---------------------------------------------------------------- primitives

/** Probability the statement holds of each image, or null where unreliable. */
export async function detectBatch(
  images: RawImage[],
  statement: string,
): Promise<Array<number | null>> {
  if (images.length === 0) return [];
  const contrast = [statement, "something else entirely", "an unclear or blank image"];
  const zsc = await warmUp();
  const out = perImage(await zsc(images, contrast, { hypothesis_template: "{}" }));
  return out.map((ranked) => {
    const ordered = contrast.map((label) => ranked.find((r) => r.label === label)?.score ?? 0);
    const probs = applyTemperature(ordered, CALIBRATION.primitives.detect.temperature);
    return floorOrUnknown(probs[0], "detect") === null ? null : probs[0];
  });
}

export async function detect(image: RawImage, statement: string): Promise<number | null> {
  return (await detectBatch([image], statement))[0];
}

export type Choice = {
  label: string;
  confidence: number;
  /** True when the best label fell below the reliability floor. */
  unknown: boolean;
  probabilities: Array<{ label: string; p: number }>;
};

/**
 * One label out of a fixed set, over many images in one call.
 *
 * Batching matters: the pipeline encodes the label set on every call, so
 * classifying 16 tiles separately would encode 102 flower names 16 times over.
 */
export async function chooseBatch(
  images: RawImage[],
  labels: readonly string[],
  hypothesis: string,
): Promise<Choice[]> {
  if (images.length === 0) return [];
  const zsc = await warmUp();
  const out = perImage(await zsc(images, labels as string[], { hypothesis_template: hypothesis }));
  return out.map((ranked) => {
    const probs = applyTemperature(
      ranked.map((r) => r.score),
      CALIBRATION.primitives.choose.temperature,
    );
    return {
      label: ranked[0].label,
      confidence: probs[0],
      unknown: !chosenReliably(probs[0], probs[1] ?? 0, labels.length),
      probabilities: ranked.map((r, i) => ({ label: r.label, p: probs[i] })),
    };
  });
}

export async function choose(
  image: RawImage,
  labels: readonly string[],
  hypothesis: string,
): Promise<Choice> {
  return (await chooseBatch([image], labels, hypothesis))[0];
}

/** Position along ordered levels, 0 = first level, 1 = last. */
export async function score(
  image: RawImage,
  levels: readonly string[],
): Promise<{ position: number; confidence: number | null }> {
  const zsc = await warmUp();
  const ranked = perImage(await zsc(image, levels as string[], { hypothesis_template: "{}" }))[0];
  const ordered = levels.map((l) => ranked.find((r) => r.label === l)?.score ?? 0);
  const probs = applyTemperature(ordered, CALIBRATION.primitives.score.temperature);
  const expected = probs.reduce((acc, p, i) => acc + p * i, 0);
  return {
    position: levels.length > 1 ? expected / (levels.length - 1) : 0,
    confidence: floorOrUnknown(Math.max(...probs), "score"),
  };
}

/** The image cut into a grid, each tile a separate look at the scene. */
async function tiles(image: RawImage): Promise<RawImage[]> {
  const tileWidth = Math.floor(image.width / GRID_COLS);
  const tileHeight = Math.floor(image.height / GRID_ROWS);
  const out: RawImage[] = [];
  for (let row = 0; row < GRID_ROWS; row++) {
    for (let col = 0; col < GRID_COLS; col++) {
      const x1 = col * tileWidth;
      const y1 = row * tileHeight;
      out.push((await image.crop([x1, y1, x1 + tileWidth, y1 + tileHeight])) as RawImage);
    }
  }
  return out;
}

/**
 * How much of the picture holds `thing`, as a distribution over tiles rather
 * than a single number.
 *
 * A point estimate would have to pick a presence threshold and then lie about
 * the marginal tiles either way.
 */
export async function coverage(
  image: RawImage,
  thing: string,
): Promise<{ tiles: RawImage[]; presence: number[]; mode: number; low: number; high: number }> {
  const grid = await tiles(image);
  const zsc = await warmUp();
  const contrast = [
    `a close-up photograph of ${thing}s`,
    "leaves, grass, soil or sky with nothing in particular",
    "an unclear or blank image",
  ];
  const out = perImage(await zsc(grid, contrast, { hypothesis_template: "{}" }));
  const presence = out.map((ranked) => {
    const raw = ranked.find((r) => r.label === contrast[0])?.score ?? 0;
    return applyTemperatureBernoulli(raw, CALIBRATION.primitives.coverage.temperature);
  });
  const { mode, low, high } = countSummary(poissonBinomial(presence));
  return { tiles: grid, presence, mode, low, high };
}

// ------------------------------------------------------------- the fact sheet

/**
 * Run the probes Jev's plan calls for and assemble what was seen.
 *
 * Nothing here interprets the question. It gathers facts; Jev draws conclusions.
 */
export async function probe(
  image: RawImage,
  plan: Plan,
  onStage: (stage: string) => void,
): Promise<FactSheet> {
  const started = performance.now();
  const vocab = VOCABULARIES[plan.vocabulary];
  const noun = plan.instanceNoun;

  onStage(`coverage "${noun}" over a ${GRID_COLS}×${GRID_ROWS} grid`);
  const { tiles: grid, presence, mode, low, high } = await coverage(image, noun);

  const floor = CALIBRATION.primitives.coverage.reliabilityFloor;
  const occupied = grid.filter((_, i) => presence[i] >= floor);

  const counts = new Map<string, { count: number; total: number }>();
  let unknownTiles = 0;

  if (occupied.length > 0) {
    onStage(`choose ×${occupied.length} over ${vocab.labels.length} ${vocab.id} labels`);
    const chosen = await chooseBatch(occupied, vocab.labels, vocab.hypothesis);
    for (const c of chosen) {
      if (c.unknown) {
        unknownTiles += 1;
        continue;
      }
      const entry = counts.get(c.label) ?? { count: 0, total: 0 };
      entry.count += 1;
      entry.total += c.confidence;
      counts.set(c.label, entry);
    }
  }

  let wholeImage: FactSheet["wholeImage"] = null;
  if (counts.size === 0) {
    onStage("no tile held the subject — reading the whole image");
    const overall = await choose(image, vocab.labels, vocab.hypothesis);
    wholeImage = overall.unknown ? null : { label: overall.label, confidence: overall.confidence };
  }

  onStage("context probes");
  const contextStatements = [
    `a photograph containing ${noun}s`,
    `several different kinds of ${noun} together`,
  ];
  const contextValues = await Promise.all(contextStatements.map((s) => detect(image, s)));
  const context = Object.fromEntries(
    contextStatements.map((s, i) => [s, contextValues[i]]),
  ) as Record<string, number | null>;

  const tallies = [...counts.entries()]
    .map(([label, v]) => ({
      label,
      count: v.count,
      meanConfidence: v.total / v.count,
      weight: v.total,
    }))
    .sort((a, b) => b.weight - a.weight);

  const totalWeight = tallies.reduce((a, t) => a + t.weight, 0);

  return {
    grid: { cols: GRID_COLS, rows: GRID_ROWS },
    tilesExamined: grid.length,
    tilesWithSubject: mode,
    tilesLow: low,
    tilesHigh: high,
    tallies,
    unknownTiles,
    wholeImage,
    context,
    dominantShare: totalWeight > 0 ? tallies[0].weight / totalWeight : null,
    imageSize: { width: image.width, height: image.height },
    elapsedMs: Math.round(performance.now() - started),
  };
}
