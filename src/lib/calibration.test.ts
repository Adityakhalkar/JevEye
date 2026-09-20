import assert from "node:assert/strict";
import { test } from "node:test";

import {
  applyTemperature,
  applyTemperatureBernoulli,
  CHOOSE_CEILING,
  CHOOSE_CHANCE_MULTIPLE,
  CHOOSE_MARGIN_RATIO,
  chosenReliably,
  countSummary,
  floorOrUnknown,
  poissonBinomial,
} from "./calibration.ts";

test("temperature 1 is the identity", () => {
  const p = [0.7, 0.2, 0.1];
  assert.deepEqual(applyTemperature(p, 1), p);
});

test("temperature above 1 softens, below 1 sharpens, and both stay normalized", () => {
  const p = [0.8, 0.15, 0.05];
  const soft = applyTemperature(p, 2);
  const sharp = applyTemperature(p, 0.5);
  assert.ok(soft[0] < p[0], "softened top probability should fall");
  assert.ok(sharp[0] > p[0], "sharpened top probability should rise");
  for (const q of [soft, sharp]) {
    assert.ok(Math.abs(q.reduce((a, b) => a + b, 0) - 1) < 1e-9);
  }
});

test("abstention returns null below the floor and the confidence above it", () => {
  assert.equal(floorOrUnknown(0.9, "detect"), 0.9);
  assert.equal(floorOrUnknown(0.1, "detect"), null);
});

test("poisson-binomial of certain detections is a point mass", () => {
  const dist = poissonBinomial([1, 1, 1]);
  assert.equal(dist.length, 4);
  assert.ok(Math.abs(dist[3] - 1) < 1e-12);
});

test("poisson-binomial is a normalized distribution and matches the binomial case", () => {
  const dist = poissonBinomial([0.5, 0.5]);
  assert.ok(Math.abs(dist.reduce((a, b) => a + b, 0) - 1) < 1e-12);
  assert.deepEqual(
    dist.map((d) => Number(d.toFixed(10))),
    [0.25, 0.5, 0.25],
  );
});

test("marginal detections widen the interval, certain ones do not", () => {
  const certain = countSummary(poissonBinomial(Array(10).fill(0.99)));
  const marginal = countSummary(poissonBinomial(Array(10).fill(0.5)));
  assert.equal(certain.mode, 10);
  assert.ok(certain.high - certain.low < marginal.high - marginal.low);
});

test("count summary covers at least the requested mass", () => {
  const dist = poissonBinomial([0.6, 0.6, 0.6, 0.6, 0.6]);
  const { low, high } = countSummary(dist, 0.9);
  const mass = dist.slice(low, high + 1).reduce((a, b) => a + b, 0);
  assert.ok(mass >= 0.9 - 1e-12, `interval held ${mass}`);
});

test("bernoulli temperature is the identity at 1 and shrinks toward 0.5 above it", () => {
  assert.equal(applyTemperatureBernoulli(0.9, 1), 0.9);
  const softened = applyTemperatureBernoulli(0.9, 2);
  assert.ok(softened < 0.9 && softened > 0.5);
  assert.ok(Math.abs(applyTemperatureBernoulli(0.5, 3) - 0.5) < 1e-12);
});

/** The bar the rule applies, restated from the constants so tuning cannot desync the tests. */
const bar = (labels: number) => Math.min(CHOOSE_CHANCE_MULTIPLE / labels, CHOOSE_CEILING);

test("the bar scales with the number of labels", () => {
  const confidence = bar(102) * 1.5;
  assert.equal(chosenReliably(confidence, 0.01, 102), true);
  // The same confidence is far below the bar when there are only a handful.
  assert.equal(chosenReliably(confidence, 0.01, 4), false);
});

test("clearing the bar is not enough without a margin over the runner-up", () => {
  const top = bar(102) * 2;
  // A runner-up closer than the ratio allows: 0.9 puts it just inside the margin.
  assert.equal(chosenReliably(top, top / (CHOOSE_MARGIN_RATIO * 0.9), 102), false);
  assert.equal(chosenReliably(top, top / (CHOOSE_MARGIN_RATIO * 2), 102), true);
});

test("the bar saturates, so a confident two-way choice still passes", () => {
  assert.equal(bar(2), CHOOSE_CEILING);
  assert.equal(chosenReliably(0.9, 0.1, 2), true);
});

test("chance-level confidence always abstains", () => {
  assert.equal(chosenReliably(1 / 102, 1 / 102, 102), false);
});
