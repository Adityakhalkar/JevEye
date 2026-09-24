/**
 * The shipped probe still performs as claimed.
 *
 * Cheap, and it guards the one artifact a refit could silently make worse: the
 * weights in `public/probes` are what the browser downloads, and their reported
 * quality is what the interface tells a visitor.
 */
import { readFileSync } from "node:fs";

const FLOOR = { accuracy: 0.9, ece: 0.05 };

export function evaluateProbe() {
  let meta;
  try {
    meta = JSON.parse(readFileSync(new URL("../public/probes/flowers.json", import.meta.url)));
  } catch {
    console.log("\nSHIPPED PROBE\n  FAIL  no probe artifact found in public/probes");
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

  console.log("\nSHIPPED PROBE");
  let failures = 0;
  for (const [name, value, ok, bound] of checks) {
    if (!ok) failures += 1;
    console.log(`  ${ok ? "pass" : "FAIL"}  ${name.padEnd(18)} ${value.toFixed(4)} (${bound})`);
  }
  console.log(`  trained on: ${meta.trained_on}`);
  return failures;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  process.exit(evaluateProbe() > 0 ? 1 : 0);
}
