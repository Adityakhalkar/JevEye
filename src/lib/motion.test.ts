import assert from "node:assert/strict";
import { test } from "node:test";

import { difference } from "./motion.ts";

const flat = (value: number, n = 32 * 24) => new Uint8Array(n).fill(value);

test("an identical picture has changed by nothing", () => {
  assert.equal(difference(flat(120), flat(120)), 0);
});

test("black against white is total change", () => {
  assert.equal(difference(flat(0), flat(255)), 1);
});

test("a small shift in brightness is a small number", () => {
  const d = difference(flat(120), flat(133));
  assert.ok(d > 0 && d < 0.06, `expected a small difference, got ${d}`);
});

test("difference is symmetric, since neither frame is privileged", () => {
  const a = flat(40);
  const b = flat(200);
  assert.equal(difference(a, b), difference(b, a));
});

test("mismatched thumbnails report total change rather than pretending", () => {
  assert.equal(difference(flat(120, 10), flat(120, 20)), 1);
  assert.equal(difference(new Uint8Array(0), new Uint8Array(0)), 1);
});

test("only part of the frame moving moves the number proportionally", () => {
  const before = flat(100);
  const after = flat(100);
  // A quarter of the pixels go fully white.
  for (let i = 0; i < after.length / 4; i++) after[i] = 255;
  const d = difference(before, after);
  assert.ok(Math.abs(d - (0.25 * 155) / 255) < 1e-9, `got ${d}`);
});
