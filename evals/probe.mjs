/**
 * The shipped probe still performs as claimed.
 *
 * Cheap, and it guards the one artifact a refit could silently make worse: the
 * weights in `public/probes` are what the browser downloads, and their reported
 * quality is what the interface tells a visitor.
 */
import { readFileSync } from "node:fs";

/**
 * Floors per probe, because the tasks are not comparable.
 *
 * Flowers are one subject photographed deliberately; COCO crops are whatever
 * happened to be in frame, at any size, partly occluded, across 80 classes that
 * include car against truck against bus. Holding both to one bar would mean
 * either excusing the first or failing the second for being harder.
 */
const FLOORS = {
  flowers: { accuracy: 0.9, ece: 0.05 },
  objects: { accuracy: 0.65, ece: 0.05 },
};

export function evaluateProbe() {
  let failures = 0;
  console.log("\nSHIPPED PROBES");
  for (const [name, FLOOR] of Object.entries(FLOORS)) {
    failures += checkOne(name, FLOOR);
  }
  return failures;
}

function checkOne(name, FLOOR) {
  let meta;
  try {
    meta = JSON.parse(readFileSync(new URL(`../public/probes/${name}.json`, import.meta.url)));
  } catch {
    console.log(`  FAIL  ${name}: no probe artifact in public/probes`);
    return 1;
  }

  const checks = [
    ["accuracy", meta.accuracy, meta.accuracy >= FLOOR.accuracy, `min ${FLOOR.accuracy}`],
    ["calibration error", meta.ece, meta.ece <= FLOOR.ece, `max ${FLOOR.ece}`],
    [
      "beats zero-shot",
      meta.accuracy - meta.zero_shot_baseline,
      meta.accuracy > meta.zero_shot_baseline,
      "must be positive",
    ],
  ];

  let failures = 0;
  for (const [label, value, ok, bound] of checks) {
    if (!ok) failures += 1;
    console.log(
      `  ${ok ? "pass" : "FAIL"}  ${name}: ${label.padEnd(18)} ${value.toFixed(4)} (${bound})`,
    );
  }
  console.log(`        ${meta.train_images ?? "?"} training images — ${meta.trained_on}`);
  return failures;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  process.exit(evaluateProbe() > 0 ? 1 : 0);
}
