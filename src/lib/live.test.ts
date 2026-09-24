import assert from "node:assert/strict";
import { test } from "node:test";

import {
  CHANGE_THRESHOLD,
  HEARTBEAT_MS,
  LiveWindow,
  MIN_JUDGMENT_GAP_MS,
  type Sample,
} from "./live.ts";

// Drift is now a plain number: 0 is an identical picture, 1 is a different one.
const STILL = 0;
const NUDGE = 0.05; // within the threshold: ordinary wobble
const MOVED = 0.4; // well past it: a different view

const sample = (at: number, drift: number, label = "dog", p = 0.6): Sample => ({
  at,
  drift,
  top: [{ label, p }],
  named: true,
});

const fill = (w: LiveWindow, from: number, drift: number, label = "dog") => {
  for (let i = 0; i < 4; i++) w.push(sample(from + i * 300, drift, label));
  return from + 3 * 300;
};

test("a still scene does not spend anything", () => {
  const w = new LiveWindow();
  let t = fill(w, 0, STILL);
  assert.equal(w.shouldJudge(t).judge, true, "the first look always happens");
  w.markJudged(t);

  // Same view, well past the rate limit but short of the heartbeat.
  t += MIN_JUDGMENT_GAP_MS + 1000;
  w.push(sample(t, STILL));
  const quiet = w.shouldJudge(t);
  assert.equal(quiet.judge, false);
  assert.equal(quiet.reason, "settled");
});

test("a moved view opens the gate", () => {
  const w = new LiveWindow();
  let t = fill(w, 0, STILL);
  w.markJudged(t);
  t += MIN_JUDGMENT_GAP_MS + 100;
  w.push(sample(t, MOVED));
  const moved = w.shouldJudge(t);
  assert.equal(moved.judge, true);
  assert.match(moved.reason, /view moved/);
});

test("a move smaller than the threshold is not a new situation", () => {
  const w = new LiveWindow();
  let t = fill(w, 0, STILL);
  w.markJudged(t);
  t += MIN_JUDGMENT_GAP_MS + 100;
  w.push(sample(t, NUDGE));
  assert.ok(NUDGE < CHANGE_THRESHOLD, "ordinary wobble sits inside the threshold");
  assert.equal(w.shouldJudge(t).judge, false);
});

test("the rate limit outranks every reason to look", () => {
  const w = new LiveWindow();
  const t = fill(w, 0, STILL);
  w.markJudged(t);
  w.push(sample(t + 100, MOVED));
  const tooSoon = w.shouldJudge(t + 100);
  assert.equal(tooSoon.judge, false);
  assert.equal(tooSoon.reason, "too soon");
});

test("a new leading label is a reason on its own", () => {
  const w = new LiveWindow();
  let t = fill(w, 0, STILL, "dog");
  w.markJudged(t);
  t += MIN_JUDGMENT_GAP_MS + 100;
  w.push(sample(t, STILL, "cat")); // identical view, different name
  const relabelled = w.shouldJudge(t);
  assert.equal(relabelled.judge, true);
  assert.match(relabelled.reason, /now leading: cat/);
});

test("an unchanging scene is still confirmed on the heartbeat", () => {
  const w = new LiveWindow();
  let t = fill(w, 0, STILL);
  w.markJudged(t);
  // Keep sampling the same still view until the heartbeat is due.
  t = fill(w, t + HEARTBEAT_MS + 100, STILL);
  assert.equal(w.shouldJudge(t).judge, true);
});

test("the window reports a direction, which no single frame can", () => {
  const w = new LiveWindow();
  // Same label throughout, confidence climbing: the shape of something approaching.
  for (let i = 0; i < 8; i++) w.push(sample(i * 300, STILL, "person", 0.2 + i * 0.08));
  const facts = w.facts(8 * 300);
  const person = facts.subjects.find((s) => s.label === "person");
  assert.ok(person);
  assert.equal(person.trend, "rising");
  assert.equal(person.seenFraction, 1);
});

test("samples older than the window are dropped", () => {
  const w = new LiveWindow();
  w.push(sample(0, STILL));
  w.push(sample(20000, STILL)); // 20s later, past the 9s window
  assert.equal(w.size, 1);
});

test("unnameable samples are counted, not silently dropped", () => {
  const w = new LiveWindow();
  w.push({ at: 0, drift: STILL, top: [{ label: "dog", p: 0.1 }], named: false });
  w.push(sample(300, STILL));
  const facts = w.facts(600);
  assert.equal(facts.unnamedSamples, 1);
  assert.equal(facts.samples, 2);
});
