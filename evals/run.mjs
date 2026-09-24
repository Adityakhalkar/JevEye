/**
 * Every measurable claim this project makes, in one command.
 *
 * Unit tests say the code does what it was written to do. These say the system
 * is as good as the README claims, which is a different question and the one
 * that quietly stops being true. A failure here is a regression in quality, not
 * a crash.
 *
 * The planner section calls the real Jev API and costs about a tenth of a cent.
 * Pass --offline to skip it.
 */
import { evaluatePlanner } from "./planner.mjs";
import { evaluateProbe } from "./probe.mjs";
import { evaluateTracker } from "./tracker.mjs";

const offline = process.argv.includes("--offline");

let failures = 0;
failures += evaluateProbe();
failures += evaluateTracker();
if (offline) {
  console.log("\nHOW JEV READS QUESTIONS\n  skipped (--offline)");
} else {
  failures += await evaluatePlanner();
}

console.log(
  failures === 0
    ? "\nAll evaluations within their thresholds.\n"
    : `\n${failures} evaluation${failures === 1 ? "" : "s"} outside threshold.\n`,
);
process.exit(failures > 0 ? 1 : 0);
