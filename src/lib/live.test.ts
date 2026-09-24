import assert from "node:assert/strict";
import { test } from "node:test";

import {
  CHANGE_THRESHOLD,
  HEARTBEAT_MS,
  LiveWindow,
  MIN_JUDGMENT_GAP_MS,
  type Sample,
} from "./live.ts";

/** Unit vectors, so similarity is the dot product, as in the real pipeline. */
const unit = (values: number[]) => {
  const norm = Math.hypot(...values);
  return new Float32Array(values.map((v) => v / norm));
};
const similarity = (a: Float32Array, b: Float32Array) =>
  a.reduce((s, x, i) => s + x * b[i], 0);

const A = unit([1, 0, 0]);
const B = unit([0.9, 0.44, 0]); // ~0.9 similar to A: a small move
const FAR = unit([0, 1, 0]); // orthogonal: a cut to something else

const sample = (at: number, embed: Float32Array, label = "dog", p = 0.6): Sample => ({
  at,
  embed,
  top: [{ label, p }],
  named: true,
});

const fill = (w: LiveWindow, from: number, embed: Float32Array, label = "dog") => {
  for (let i = 0; i < 4; i++) w.push(sample(from + i * 300, embed, label));
  return from + 3 * 300;
};

test("a still scene does not spend anything", () => {
  const w = new LiveWindow();
  let t = fill(w, 0, A);
  assert.equal(w.shouldJudge(t, similarity).judge, true, "the first look always happens");
  w.markJudged(t);

  // Same view, well past the rate limit but short of the heartbeat.
  t += MIN_JUDGMENT_GAP_MS + 1000;
  w.push(sample(t, A));
  const quiet = w.shouldJudge(t, similarity);
  assert.equal(quiet.judge, false);
  assert.equal(quiet.reason, "settled");
});

test("a moved view opens the gate", () => {
  const w = new LiveWindow();
  let t = fill(w, 0, A);
  w.markJudged(t);
  t += MIN_JUDGMENT_GAP_MS + 100;
  w.push(sample(t, FAR));
  const moved = w.shouldJudge(t, similarity);
  assert.equal(moved.judge, true);
  assert.match(moved.reason, /view moved/);
});

test("a move smaller than the threshold is not a new situation", () => {
  const w = new LiveWindow();
  let t = fill(w, 0, A);
  w.markJudged(t);
  t += MIN_JUDGMENT_GAP_MS + 100;
  w.push(sample(t, B));
  assert.ok(1 - similarity(A, B) < CHANGE_THRESHOLD, "B is within the threshold of A");
  assert.equal(w.shouldJudge(t, similarity).judge, false);
});

test("the rate limit outranks every reason to look", () => {
  const w = new LiveWindow();
  const t = fill(w, 0, A);
  w.markJudged(t);
  w.push(sample(t + 100, FAR));
  const tooSoon = w.shouldJudge(t + 100, similarity);
  assert.equal(tooSoon.judge, false);
  assert.equal(tooSoon.reason, "too soon");
});

test("a new leading label is a reason on its own", () => {
  const w = new LiveWindow();
  let t = fill(w, 0, A, "dog");
  w.markJudged(t);
  t += MIN_JUDGMENT_GAP_MS + 100;
  w.push(sample(t, A, "cat")); // identical view, different name
  const relabelled = w.shouldJudge(t, similarity);
  assert.equal(relabelled.judge, true);
  assert.match(relabelled.reason, /now leading: cat/);
});

test("an unchanging scene is still confirmed on the heartbeat", () => {
  const w = new LiveWindow();
  let t = fill(w, 0, A);
  w.markJudged(t);
  // Keep sampling the same still view until the heartbeat is due.
  t = fill(w, t + HEARTBEAT_MS + 100, A);
  assert.equal(w.shouldJudge(t, similarity).judge, true);
});

test("the window reports a direction, which no single frame can", () => {
  const w = new LiveWindow();
  // Same label throughout, confidence climbing: the shape of something approaching.
  for (let i = 0; i < 8; i++) w.push(sample(i * 300, A, "person", 0.2 + i * 0.08));
  const facts = w.facts(8 * 300, similarity);
  const person = facts.subjects.find((s) => s.label === "person");
  assert.ok(person);
  assert.equal(person.trend, "rising");
  assert.equal(person.seenFraction, 1);
});

test("samples older than the window are dropped", () => {
  const w = new LiveWindow();
  w.push(sample(0, A));
  w.push(sample(20000, A)); // 20s later, past the 9s window
  assert.equal(w.size, 1);
});

test("unnameable samples are counted, not silently dropped", () => {
  const w = new LiveWindow();
  w.push({ at: 0, embed: A, top: [{ label: "dog", p: 0.1 }], named: false });
  w.push(sample(300, A));
  const facts = w.facts(600, similarity);
  assert.equal(facts.unnamedSamples, 1);
  assert.equal(facts.samples, 2);
});
