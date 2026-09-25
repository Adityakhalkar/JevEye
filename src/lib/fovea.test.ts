import assert from "node:assert/strict";
import { test } from "node:test";

import { Doubt, FOVEA_SHARE, outward, type Held } from "./fovea.ts";

const FRAME = { width: 336, height: 252 };
const box = (x: number, y: number, w = 40, h = 40) => ({ x1: x, y1: y, x2: x + w, y2: y + h });
const middle = (b: ReturnType<typeof box>) => ({ x: (b.x1 + b.x2) / 2, y: (b.y1 + b.y2) / 2 });
const holds = (region: ReturnType<typeof box>, point: { x: number; y: number }) =>
  point.x >= region.x1 && point.x <= region.x2 && point.y >= region.y1 && point.y <= region.y2;
const held = (b: ReturnType<typeof box>, lock: number, lockable = true): Held => ({
  box: b,
  lock,
  lockable,
});

test("the close look goes where the tracker is least sure", () => {
  const doubt = new Doubt();
  const struggling = box(260, 190);
  // Enough passes that doubt outweighs plain curiosity about elsewhere.
  for (let i = 1; i <= 10; i++) {
    doubt.observe([held(box(20, 20), 0.95), held(struggling, 0.1)], FRAME, i * 100);
  }
  assert.ok(
    holds(doubt.where(FRAME, 1000), middle(struggling)),
    "it looked away from the thing it could not hold",
  );
});

test("something too small to recognise is the strongest call for a closer look", () => {
  const doubt = new Doubt();
  const tiny = box(260, 190);
  for (let i = 1; i <= 10; i++) {
    doubt.observe([held(box(20, 20), 0.6), held(tiny, 0.6, false)], FRAME, i * 100);
  }
  assert.ok(holds(doubt.where(FRAME, 1000), middle(tiny)));
});

test("having looked somewhere, it looks elsewhere next", () => {
  const doubt = new Doubt();
  const trouble = box(280, 200);
  for (let i = 1; i <= 10; i++) doubt.observe([held(trouble, 0, false)], FRAME, i * 100);
  const first = doubt.where(FRAME, 1000);
  assert.ok(holds(first, middle(trouble)));
  doubt.looked(first, FRAME, 1000);
  assert.ok(
    !holds(doubt.where(FRAME, 1100), middle(trouble)),
    "it fixated on a region it had just examined",
  );
});

test("a region that keeps disappointing does not keep winning", () => {
  const doubt = new Doubt();
  const hopeless = box(280, 200);
  const regions = new Set<string>();
  // The same unrecognisable subject, looked at again and again.
  for (let pass = 0; pass < 6; pass++) {
    const now = pass * 1500;
    doubt.observe([held(hopeless, 0, false)], FRAME, now);
    const region = doubt.where(FRAME, now);
    regions.add(`${Math.round(region.x1)},${Math.round(region.y1)}`);
    doubt.looked(region, FRAME, now);
  }
  assert.ok(regions.size > 1, "the close look never went anywhere else");
});

test("nowhere stays uninteresting: somewhere unexamined eventually wins", () => {
  const doubt = new Doubt();
  const trouble = box(280, 200);
  for (let i = 1; i <= 10; i++) doubt.observe([held(trouble, 0.2)], FRAME, i * 100);
  const region = doubt.where(FRAME, 1000);
  doubt.looked(region, FRAME, 1000);
  // Nothing new is reported anywhere; the far corner has simply been ignored.
  const later = doubt.where(FRAME, 40_000);
  assert.ok(!holds(later, middle(trouble)) || doubt.unsure() > 0);
  assert.ok(later.x1 >= 0 && later.y1 >= 0);
});

test("a close look covers a share of the frame and stays inside it", () => {
  const doubt = new Doubt();
  for (let i = 1; i <= 5; i++) doubt.observe([held(box(0, 0, 10, 10), 0, false)], FRAME, i * 100);
  const region = doubt.where(FRAME, 500);
  assert.ok(region.x1 >= 0 && region.y1 >= 0);
  assert.ok(region.x2 <= FRAME.width + 1e-6 && region.y2 <= FRAME.height + 1e-6);
  assert.ok(Math.abs(region.x2 - region.x1 - FRAME.width * FOVEA_SHARE) < 1e-6);
});

test("doubt is about being unsure lately, not a tally of every difficulty", () => {
  const doubt = new Doubt();
  for (let i = 1; i <= 10; i++) doubt.observe([held(box(280, 200), 0, false)], FRAME, i * 100);
  const troubled = doubt.unsure();
  // Ten seconds later, with everything now held perfectly.
  doubt.observe([held(box(280, 200), 1)], FRAME, 11_000);
  assert.ok(doubt.unsure() < troubled * 0.1, `doubt lingered at ${doubt.unsure().toFixed(2)}`);
});

test("a box found in a close look comes back in the frame's own coordinates", () => {
  const region = { x1: 100, y1: 50, x2: 200, y2: 125 };
  const looked = { width: 400, height: 300 };
  const back = outward({ x1: 190, y1: 140, x2: 210, y2: 160 }, region, looked);
  assert.ok(Math.abs((back.x1 + back.x2) / 2 - 150) < 0.5);
  assert.ok(Math.abs((back.y1 + back.y2) / 2 - 87.5) < 0.5);
  const whole = outward({ x1: 0, y1: 0, x2: 400, y2: 300 }, region, looked);
  assert.deepEqual(
    [whole.x1, whole.y1, whole.x2, whole.y2],
    [region.x1, region.y1, region.x2, region.y2],
  );
});
