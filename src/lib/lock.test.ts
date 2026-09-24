import assert from "node:assert/strict";
import { test } from "node:test";

import { correlate, print, relock, type Print } from "./lock.ts";
import { asCanvas, boxAt, canvasOf, emptyScene, SIZE, sceneWith, type Stub } from "./scene.mock.ts";

const scratch = canvasOf(24, 24, () => 0);
const printOf = (stub: Stub, rect: ReturnType<typeof boxAt>) =>
  print(asCanvas(stub), rect, asCanvas(scratch));
const relockIn = (stub: Stub, template: Print, expected: ReturnType<typeof boxAt>) =>
  relock(asCanvas(stub), template, expected, asCanvas(scratch));

test("a print of a flat region is refused, because it would match anything", () => {
  assert.equal(printOf(canvasOf(320, 240, () => 128), boxAt(10, 10)), null);
});

test("a print correlates perfectly with itself", () => {
  const p = printOf(sceneWith(100, 80), boxAt(100, 80));
  assert.ok(p);
  assert.ok(Math.abs(correlate(p, p) - 1) < 1e-6);
});

test("brightening the whole scene does not change the print", () => {
  const dim = printOf(sceneWith(100, 80), boxAt(100, 80)) as Print;
  const bright = printOf(sceneWith(100, 80, 25), boxAt(100, 80)) as Print;
  assert.ok(correlate(dim, bright) > 0.99, `exposure changed the print: ${correlate(dim, bright)}`);
});

const template = printOf(sceneWith(100, 80), boxAt(100, 80)) as Print;

// Every distance the subject can move and still be inside the search window.
for (const [dx, dy] of [
  [0, 0],
  [2, 1],
  [6, 4],
  [-7, 5],
  [18, 10],
  [23, -15],
] as const) {
  test(`the lock follows the subject ${dx},${dy} pixels and lands within a pixel`, () => {
    const found = relockIn(sceneWith(100 + dx, 80 + dy), template, boxAt(100, 80));
    assert.ok(found, "the subject was found");
    assert.ok(found.score > 0.7, `weak match: ${found.score.toFixed(3)}`);
    const off = Math.hypot(found.rect.x1 - (100 + dx), found.rect.y1 - (80 + dy));
    assert.ok(off < 1, `landed ${off.toFixed(2)}px away, which would look like a jump`);
  });
}

test("the box keeps the size it was given, so the lock only moves it", () => {
  const found = relockIn(sceneWith(112, 86), template, boxAt(100, 80));
  assert.ok(found);
  assert.ok(Math.abs(found.rect.x2 - found.rect.x1 - SIZE) < 1e-6);
  assert.ok(Math.abs(found.rect.y2 - found.rect.y1 - SIZE) < 1e-6);
});

test("when the subject is gone there is nothing to lock on to", () => {
  const found = relockIn(emptyScene(), template, boxAt(100, 80));
  assert.ok(!found, `claimed a match of ${found?.score.toFixed(3)} on an empty field`);
});

test("noise does not pass as the subject", () => {
  let seed = 1;
  const random = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff) * 255;
  const found = relockIn(canvasOf(320, 240, random), template, boxAt(100, 80));
  assert.ok(!found || found.score < 0.5, `noise scored ${found?.score.toFixed(3)}`);
});

test("a contrast-inverted subject scores clearly below the real one", () => {
  const real = sceneWith(100, 80);
  const inverted = canvasOf(320, 240, (px, py) => 255 - real.buffer[py * 320 + px]);
  const honest = relockIn(real, template, boxAt(100, 80));
  const impostor = relockIn(inverted, template, boxAt(100, 80));
  assert.ok(honest);
  assert.ok(
    !impostor || impostor.score < honest.score - 0.1,
    `impostor ${impostor?.score.toFixed(3)} vs real ${honest.score.toFixed(3)}`,
  );
});

test("a subject that moved further than the search window is not chased", () => {
  const found = relockIn(sceneWith(220, 80), template, boxAt(100, 80));
  assert.ok(!found || found.score < 0.7, `chased it to ${found?.rect.x1.toFixed(0)}`);
});
