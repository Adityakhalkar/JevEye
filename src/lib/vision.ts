"use client";

/**
 * The vision layer. Answers questions of fact about pixels, with a calibrated
 * confidence or an abstention. It never judges meaning: that is Jev's job, and
 * this module never calls Jev.
 *
 * Everything here runs in the browser. The image is never uploaded.
 *
 * CLIP's two towers are loaded separately rather than through a pipeline, so an
 * image's 512-d embedding is available directly. That is what lets a trained
 * linear probe and the zero-shot text classifier share one forward pass — and
 * it lets text embeddings be cached, which a pipeline re-computes every call.
 *
 * Instance detection was the original plan, but OWL-ViT's `class_head` Cast node
 * has no ONNX Runtime Web implementation at q8, q4f16 or fp16, and only its
 * 583 MB fp32 graph could load. So JevEye reports how much of the picture each
 * kind covers rather than how many of them there are.
 */
import {
  AutoProcessor,
  AutoTokenizer,
  CLIPTextModelWithProjection,
  CLIPVisionModelWithProjection,
  RawImage,
  env,
} from "@huggingface/transformers";

import {
  CALIBRATION,
  applyTemperature,
  applyTemperatureBernoulli,
  chosenReliably,
  countSummary,
  floorOrUnknown,
  poissonBinomial,
} from "./calibration";
import { SCALES } from "./scales";
import { IDENTIFYING, type FactSheet, type Plan } from "./types";
import { VOCABULARIES, type Vocabulary } from "./vocab";

env.allowLocalModels = false;

const MODEL = "Xenova/clip-vit-base-patch32";
const DTYPE = "q8";
const DIM = 512;
/** The scale CLIP learned during training; it makes cosines usable, not honest. */
const CLIP_LOGIT_SCALE = 100;

const LONGEST_EDGE = 1280;

/**
 * ponytail: a fixed non-overlapping 4×4 grid. Coarse — a flower straddling two
 * tiles is seen twice, and one smaller than a tile's detail is diluted by its
 * background. Upgrade path when it matters: overlapping tiles at two scales.
 */
const GRID_COLS = 4;
const GRID_ROWS = 4;

export type LoadProgress = { file: string; percent: number };

type Loaded = {
  processor: Awaited<ReturnType<typeof AutoProcessor.from_pretrained>>;
  tokenizer: Awaited<ReturnType<typeof AutoTokenizer.from_pretrained>>;
  vision: Awaited<ReturnType<typeof CLIPVisionModelWithProjection.from_pretrained>>;
  text: Awaited<ReturnType<typeof CLIPTextModelWithProjection.from_pretrained>>;
};

let loading: Promise<Loaded> | null = null;
const textCache = new Map<string, Float32Array>();

function device(): "webgpu" | "wasm" {
  return typeof navigator !== "undefined" && "gpu" in navigator ? "webgpu" : "wasm";
}

/** Both towers, loaded once and cached by the browser thereafter. */
export async function warmUp(onProgress?: (p: LoadProgress) => void): Promise<Loaded> {
  const progress_callback = (event: { status?: string; file?: string; progress?: number }) => {
    if (event.status === "progress" && event.file) {
      onProgress?.({ file: event.file, percent: Math.round(event.progress ?? 0) });
    }
  };
  loading ??= (async () => {
    const options = { dtype: DTYPE, device: device(), progress_callback } as never;
    const [processor, tokenizer, vision, text] = await Promise.all([
      AutoProcessor.from_pretrained(MODEL, { progress_callback } as never),
      AutoTokenizer.from_pretrained(MODEL, { progress_callback } as never),
      CLIPVisionModelWithProjection.from_pretrained(MODEL, options),
      CLIPTextModelWithProjection.from_pretrained(MODEL, options),
    ]);
    return { processor, tokenizer, vision, text } as Loaded;
  })();
  return loading;
}

function normalize(row: number[]): Float32Array {
  const out = new Float32Array(DIM);
  let sum = 0;
  for (const x of row) sum += x * x;
  const norm = Math.sqrt(sum) || 1;
  for (let i = 0; i < DIM; i++) out[i] = row[i] / norm;
  return out;
}

/** One 512-d unit vector per image. Every primitive is built on this. */
export async function embedImages(images: RawImage[]): Promise<Float32Array[]> {
  if (images.length === 0) return [];
  const { processor, vision } = await warmUp();
  const inputs = await (processor as unknown as (i: RawImage[]) => Promise<unknown>)(images);
  const { image_embeds } = await (vision as (i: unknown) => Promise<{ image_embeds: { tolist(): number[][] } }>)(inputs);
  return image_embeds.tolist().map(normalize);
}

/** One 512-d unit vector per sentence, cached because prompts repeat. */
export async function embedTexts(prompts: string[]): Promise<Float32Array[]> {
  const missing = [...new Set(prompts.filter((p) => !textCache.has(p)))];
  if (missing.length > 0) {
    const { tokenizer, text } = await warmUp();
    // The tokenizer is synchronous, unlike the image processor.
    const inputs = (tokenizer as unknown as (p: string[], o: object) => unknown)(missing, {
      padding: true,
      truncation: true,
    });
    const { text_embeds } = await (text as (i: unknown) => Promise<{ text_embeds: { tolist(): number[][] } }>)(inputs);
    text_embeds.tolist().forEach((row, i) => textCache.set(missing[i], normalize(row)));
  }
  return prompts.map((p) => textCache.get(p)!);
}

const dot = (a: Float32Array, b: Float32Array) => {
  let s = 0;
  for (let i = 0; i < DIM; i++) s += a[i] * b[i];
  return s;
};

const softmax = (logits: number[]) => {
  const max = Math.max(...logits);
  const e = logits.map((z) => Math.exp(z - max));
  const sum = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / sum);
};

/** Score images against sentences — the zero-shot classifier, open vocabulary. */
async function scoreAgainst(images: RawImage[], prompts: string[]): Promise<number[][]> {
  const [embeds, texts] = await Promise.all([embedImages(images), embedTexts(prompts)]);
  return embeds.map((e) => softmax(texts.map((t) => dot(e, t) * CLIP_LOGIT_SCALE)));
}

// ------------------------------------------------------------------- probes

/** A linear probe fitted offline on frozen embeddings; see `tools/fit.py`. */
export type Probe = {
  weights: Float32Array;
  bias: Float32Array;
  classes: number;
  /** Fitted on a held-out split, unlike the zero-shot path. */
  temperature: number;
  /** Measured on images neither the probe nor the temperature ever saw. */
  accuracy: number;
  ece: number;
};

const probeCache = new Map<string, Promise<Probe | null>>();

async function loadProbe(vocab: Vocabulary): Promise<Probe | null> {
  if (!vocab.probe) return null;
  const base = vocab.probe;
  let cached = probeCache.get(base);
  if (!cached) {
    cached = (async () => {
      try {
        const [meta, weights, bias] = await Promise.all([
          fetch(`${base}.json`).then((r) => r.json()),
          fetch(`${base}.w.bin`).then((r) => r.arrayBuffer()),
          fetch(`${base}.b.bin`).then((r) => r.arrayBuffer()),
        ]);
        return {
          weights: new Float32Array(weights),
          bias: new Float32Array(bias),
          classes: meta.classes,
          temperature: meta.temperature,
          accuracy: meta.accuracy,
          ece: meta.ece,
        };
      } catch {
        // A missing probe is not an error: the zero-shot path still works.
        return null;
      }
    })();
    probeCache.set(base, cached);
  }
  return cached;
}

function probeProbabilities(embed: Float32Array, probe: Probe): number[] {
  const logits = new Array(probe.classes);
  for (let c = 0; c < probe.classes; c++) {
    let s = probe.bias[c];
    const row = c * DIM;
    for (let i = 0; i < DIM; i++) s += probe.weights[row + i] * embed[i];
    logits[c] = s / probe.temperature;
  }
  return softmax(logits);
}

// ---------------------------------------------------------------- primitives

/** Probability the statement holds of each image, or null where unreliable. */
export async function detectBatch(
  images: RawImage[],
  statement: string,
): Promise<Array<number | null>> {
  if (images.length === 0) return [];
  const contrast = [statement, "something else entirely", "an unclear or blank image"];
  const scored = await scoreAgainst(images, contrast);
  return scored.map((probs) => {
    const tempered = applyTemperature(probs, CALIBRATION.primitives.detect.temperature);
    return floorOrUnknown(tempered[0], "detect") === null ? null : tempered[0];
  });
}

export async function detect(image: RawImage, statement: string): Promise<number | null> {
  return (await detectBatch([image], statement))[0];
}

export type Choice = {
  label: string;
  confidence: number;
  /** True when the best label fell below the reliability bar. */
  unknown: boolean;
  /** Whether a fitted probe answered, or the zero-shot text classifier did. */
  source: "probe" | "zero-shot";
  /** The probe's measured quality, when a probe answered. */
  quality: { accuracy: number; ece: number } | null;
  probabilities: Array<{ label: string; p: number }>;
};

/**
 * One label out of a vocabulary, for many images in one pass.
 *
 * Uses the vocabulary's trained probe when there is one, and the zero-shot text
 * classifier otherwise. Both read the same embedding, so the choice costs one
 * forward pass either way.
 */
export async function chooseBatch(images: RawImage[], vocab: Vocabulary): Promise<Choice[]> {
  if (images.length === 0) return [];
  const [embeds, probe] = await Promise.all([embedImages(images), loadProbe(vocab)]);

  let perImage: number[][];
  let source: Choice["source"];
  let quality: Choice["quality"] = null;
  if (probe && probe.classes === vocab.labels.length) {
    perImage = embeds.map((e) => probeProbabilities(e, probe));
    source = "probe";
    quality = { accuracy: probe.accuracy, ece: probe.ece };
  } else {
    const texts = await embedTexts(vocab.labels.map((l) => vocab.hypothesis.replace("{}", l)));
    perImage = embeds.map((e) =>
      applyTemperature(
        softmax(texts.map((t) => dot(e, t) * CLIP_LOGIT_SCALE)),
        CALIBRATION.primitives.choose.temperature,
      ),
    );
    source = "zero-shot";
  }

  return perImage.map((probs) => {
    const ranked = vocab.labels
      .map((label, i) => ({ label, p: probs[i] }))
      .sort((a, b) => b.p - a.p);
    return {
      label: ranked[0].label,
      confidence: ranked[0].p,
      unknown: !chosenReliably(ranked[0].p, ranked[1]?.p ?? 0, vocab.labels.length),
      source,
      quality,
      probabilities: ranked,
    };
  });
}

export async function choose(image: RawImage, vocab: Vocabulary): Promise<Choice> {
  return (await chooseBatch([image], vocab))[0];
}

/** Position along ordered levels, 0 = first level, 1 = last. */
export async function score(
  image: RawImage,
  levels: readonly string[],
): Promise<{ position: number; confidence: number | null }> {
  const raw = (await scoreAgainst([image], levels as string[]))[0];
  const probs = applyTemperature(raw, CALIBRATION.primitives.score.temperature);
  const expected = probs.reduce((acc, p, i) => acc + p * i, 0);
  return {
    position: levels.length > 1 ? expected / (levels.length - 1) : 0,
    confidence: floorOrUnknown(Math.max(...probs), "score"),
  };
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
 * How much of the picture holds `thing`, as a distribution over tiles.
 *
 * A point estimate would have to pick a presence threshold and then lie about
 * the marginal tiles either way.
 */
export async function coverage(
  image: RawImage,
  thing: string,
): Promise<{ tiles: RawImage[]; presence: number[]; mode: number; low: number; high: number }> {
  const grid = await tiles(image);
  const contrast = [
    `a close-up photograph of ${thing}s`,
    "leaves, grass, soil or sky with nothing in particular",
    "an unclear or blank image",
  ];
  const scored = await scoreAgainst(grid, contrast);
  const presence = scored.map((probs) =>
    applyTemperatureBernoulli(probs[0], CALIBRATION.primitives.coverage.temperature),
  );
  const { mode, low, high } = countSummary(poissonBinomial(presence));
  return { tiles: grid, presence, mode, low, high };
}

// ------------------------------------------------------------- the fact sheet

/**
 * Run the probes the reading calls for and assemble what was seen.
 *
 * The reading Jev picked decides which probes run, not just how the answer is
 * phrased. Nothing here interprets the question — it gathers facts, and Jev
 * draws the conclusion.
 */
export async function probe(
  image: RawImage,
  plan: Plan,
  onStage: (stage: string) => void,
): Promise<FactSheet> {
  const started = performance.now();
  const noun = plan.instanceNoun;

  const contextProbes = async (statements: string[]) => {
    onStage("context probes");
    const values = await Promise.all(statements.map((t) => detect(image, t)));
    return Object.fromEntries(statements.map((t, i) => [t, values[i]])) as Record<
      string,
      number | null
    >;
  };

  const common = () => ({
    imageSize: { width: image.width, height: image.height },
    elapsedMs: Math.round(performance.now() - started),
  });

  // ---- rating: one ordered scale over the whole image, no vocabulary at all
  if (plan.reading === "rating" && plan.scale) {
    const scale = SCALES[plan.scale];
    onStage(`score the "${scale.id}" scale over ${scale.levels.length} levels`);
    const { position, confidence } = await score(image, scale.levels);
    const context = await contextProbes([
      "a clear photograph of a single obvious subject",
      `a photograph where the ${scale.id} of the subject is easy to judge`,
    ]);
    return {
      kind: "rating",
      scale: scale.id,
      levels: scale.levels,
      position,
      confidence,
      context,
      ...common(),
    };
  }

  onStage(`coverage "${noun}" over a ${GRID_COLS}×${GRID_ROWS} grid`);
  const { tiles: grid, presence, mode, low, high } = await coverage(image, noun);
  const coverageFacts = {
    grid: { cols: GRID_COLS, rows: GRID_ROWS },
    tilesExamined: grid.length,
    tilesWithSubject: mode,
    tilesLow: low,
    tilesHigh: high,
  };
  const floor = CALIBRATION.primitives.coverage.reliabilityFloor;
  const occupied = grid.filter((_, i) => presence[i] >= floor);

  // ---- count: coverage is the answer, so skip the classifier entirely
  if (plan.reading === "count") {
    const context = await contextProbes([
      `a photograph containing ${noun}s`,
      `a photograph containing a single ${noun}`,
    ]);
    return { kind: "count", noun, ...coverageFacts, context, ...common() };
  }

  // ---- presence: one open-vocabulary statement about a subject Jev chose
  if (plan.reading === "presence" && plan.subject) {
    const statement = `a photograph containing a ${plan.subject}`;
    onStage(`detect "${statement}"`);
    const probability = await detect(image, statement);

    let tilesMatchingSubject = 0;
    if (occupied.length > 0) {
      onStage(`checking ${occupied.length} tiles for "${plan.subject}"`);
      const perTile = await detectBatch(occupied, statement);
      tilesMatchingSubject = perTile.filter((p) => p !== null).length;
    }

    const context = await contextProbes([
      `a photograph containing ${noun}s`,
      "an unclear, cluttered or ambiguous photograph",
    ]);
    return {
      kind: "presence",
      subject: plan.subject,
      statement,
      probability,
      tilesMatchingSubject,
      tilesChecked: occupied.length,
      ...coverageFacts,
      context,
      ...common(),
    };
  }

  // ---- identify: name the kinds, tile by tile
  const vocab = VOCABULARIES[plan.vocabulary];
  const counts = new Map<string, { count: number; total: number }>();
  let unknownTiles = 0;
  let source: Choice["source"] = "zero-shot";
  let quality: Choice["quality"] = null;

  if (occupied.length > 0) {
    onStage(`choose ×${occupied.length} over ${vocab.labels.length} ${vocab.id} labels`);
    const chosen = await chooseBatch(occupied, vocab);
    source = chosen[0]?.source ?? source;
    quality = chosen[0]?.quality ?? null;
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

  let wholeImage: { label: string; confidence: number } | null = null;
  if (counts.size === 0) {
    onStage("no tile held the subject — reading the whole image");
    const overall = await choose(image, vocab);
    source = overall.source;
    quality = overall.quality;
    wholeImage = overall.unknown ? null : { label: overall.label, confidence: overall.confidence };
  }

  const context = await contextProbes([
    `a photograph containing ${noun}s`,
    `several different kinds of ${noun} together`,
  ]);

  const tallies = [...counts.entries()]
    .map(([label, v]) => ({
      label,
      count: v.count,
      meanConfidence: v.total / v.count,
      weight: v.total,
    }))
    .sort((a, b) => b.weight - a.weight);
  const totalWeight = tallies.reduce((a, t) => a + t.weight, 0);

  if (!IDENTIFYING.has(plan.reading)) {
    onStage(`reading "${plan.reading}" fell back to naming kinds`);
  }

  return {
    kind: "identify",
    tallies,
    unknownTiles,
    wholeImage,
    classifier: source,
    classifierAccuracy: quality?.accuracy ?? null,
    classifierEce: quality?.ece ?? null,
    dominantShare: totalWeight > 0 ? tallies[0].weight / totalWeight : null,
    ...coverageFacts,
    context,
    ...common(),
  };
}
