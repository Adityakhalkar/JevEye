/**
 * Calibration turns a model's raw confidence into a number that means what it says.
 *
 * Two knobs per primitive: a temperature that rescales the probabilities, and a
 * reliability floor below which the primitive abstains instead of guessing.
 *
 * IMPORTANT: the temperatures below are NOT fitted. They are identity (1.0)
 * placeholders, so probabilities here are raw CLIP/OWL-ViT confidences and are
 * very likely overconfident. Fitting them needs a labelled held-out split
 * (COCO val, Flowers-102 test) and an offline run that has not happened yet.
 * The floors are conservative hand-set values, not measured operating points.
 */
export type PrimitiveName = "detect" | "choose" | "score" | "coverage";

export type PrimitiveCalibration = {
  /** Divides the logits before re-normalizing. >1 softens, <1 sharpens. */
  temperature: number;
  /** Confidence below which the primitive returns `unknown`. */
  reliabilityFloor: number;
};

export type Calibration = {
  fitted: boolean;
  note: string;
  primitives: Record<PrimitiveName, PrimitiveCalibration>;
};

export const CALIBRATION: Calibration = {
  fitted: false,
  note:
    "Unfitted identity temperatures. Confidences are raw model outputs and are probably overconfident.",
  primitives: {
    detect: { temperature: 1.0, reliabilityFloor: 0.55 },
    choose: { temperature: 1.0, reliabilityFloor: 0.35 },
    score: { temperature: 1.0, reliabilityFloor: 0.3 },
    coverage: { temperature: 1.0, reliabilityFloor: 0.45 },
  },
};

/**
 * Re-softmax a probability vector at temperature `t`.
 *
 * Safe to apply to probabilities rather than logits: softmax(z/t) is recoverable
 * from p because log p = z - logZ, and the constant cancels in the softmax.
 */
export function applyTemperature(probs: number[], t: number): number[] {
  if (t === 1 || probs.length === 0) return probs.slice();
  const logits = probs.map((p) => Math.log(Math.max(p, 1e-12)) / t);
  const max = Math.max(...logits);
  const exp = logits.map((z) => Math.exp(z - max));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map((e) => e / sum);
}

/** `null` means the primitive abstained: the answer is unknown, not 50/50. */
export function floorOrUnknown(confidence: number, p: PrimitiveName): number | null {
  return confidence >= CALIBRATION.primitives[p].reliabilityFloor ? confidence : null;
}

/**
 * How far above chance a label must sit, and how far clear of the runner-up.
 *
 * Hand-set, not fitted. 4x chance let labels at 0.09 through on a 102-way
 * choice, which is noise; 12x keeps the signal and drops them. This is exactly
 * the knob a fitted calibration would set from a labelled split, and choosing it
 * by eye on one photograph is not validation.
 */
export const CHOOSE_CHANCE_MULTIPLE = 12;
export const CHOOSE_MARGIN_RATIO = 1.5;
/** The chance multiple has to saturate, or a two-way choice could never pass. */
export const CHOOSE_CEILING = 0.5;

/**
 * Whether a choice over `labelCount` labels is worth reporting.
 *
 * A flat floor cannot serve both an 80-way and a 102-way choice: the same 0.3
 * is lax for two labels and unreachable for a hundred, where even a correct
 * top-1 is diffuse. So the test is relative — comfortably above chance, and
 * clear of the runner-up — which is what "this label, not that one" means.
 */
export function chosenReliably(top: number, runnerUp: number, labelCount: number): boolean {
  const chance = 1 / labelCount;
  const bar = Math.min(CHOOSE_CHANCE_MULTIPLE * chance, CHOOSE_CEILING);
  return top >= bar && top >= CHOOSE_MARGIN_RATIO * runnerUp;
}

/**
 * Exact distribution over the number of true positives among independent
 * observations with probabilities `ps` (a Poisson-binomial).
 *
 * This is why counting needs no confidence threshold: marginal observations
 * widen the distribution instead of being silently included or dropped.
 */
export function poissonBinomial(ps: number[]): number[] {
  let dist = [1];
  for (const p of ps) {
    const next = new Array(dist.length + 1).fill(0);
    for (let k = 0; k < dist.length; k++) {
      next[k] += dist[k] * (1 - p);
      next[k + 1] += dist[k] * p;
    }
    dist = next;
  }
  return dist;
}

/** Most likely count, and the narrowest interval holding at least `mass`. */
export function countSummary(dist: number[], mass = 0.9): { mode: number; low: number; high: number } {
  let mode = 0;
  for (let k = 1; k < dist.length; k++) if (dist[k] > dist[mode]) mode = k;
  let low = mode;
  let high = mode;
  let total = dist[mode] ?? 0;
  while (total < mass && (low > 0 || high < dist.length - 1)) {
    const down = low > 0 ? dist[low - 1] : -1;
    const up = high < dist.length - 1 ? dist[high + 1] : -1;
    if (up >= down) total += dist[++high];
    else total += dist[--low];
  }
  return { mode, low, high };
}

/**
 * Temperature for an independent Bernoulli confidence, such as one detection box.
 *
 * Detection scores are per-box sigmoids, not a distribution over boxes, so they
 * must be rescaled one at a time rather than re-softmaxed against each other.
 */
export function applyTemperatureBernoulli(p: number, t: number): number {
  if (t === 1) return p;
  const clamped = Math.min(Math.max(p, 1e-12), 1 - 1e-12);
  const z = Math.log(clamped / (1 - clamped)) / t;
  return 1 / (1 + Math.exp(-z));
}
