import assert from "node:assert/strict";
import { test } from "node:test";

import {
  IOU_GATE,
  MAX_MISSES,
  MIN_HITS,
  Tracker,
  centre,
  gateFor,
  heading,
  iou,
  predictBox,
  type Box,
} from "./track.ts";

const box = (x: number, y: number, w = 100, h = 100): Box => ({
  x1: x,
  y1: y,
  x2: x + w,
  y2: y + h,
});
const seen = (label: string, b: Box, score = 0.9) => ({ label, score, box: b });

test("identical boxes overlap completely, disjoint ones not at all", () => {
  assert.equal(iou(box(0, 0), box(0, 0)), 1);
  assert.equal(iou(box(0, 0), box(500, 500)), 0);
});

test("a half-overlapping pair scores a third, which is the union's fault", () => {
  // Two 100x100 boxes sharing 50x100: intersection 5000, union 15000.
  assert.ok(Math.abs(iou(box(0, 0), box(50, 0)) - 1 / 3) < 1e-9);
});

test("a track is provisional until it has been seen enough times", () => {
  const t = new Tracker(640 * 480);
  assert.equal(t.update([seen("dog", box(10, 10))], 0).length, 0, "one sighting is not a track");
  const confirmed = t.update([seen("dog", box(12, 10))], 300);
  assert.equal(confirmed.length, MIN_HITS - 1);
  assert.equal(confirmed[0].label, "dog");
});

test("the same object keeps one identity across frames", () => {
  const t = new Tracker(640 * 480);
  t.update([seen("dog", box(10, 10))], 0);
  const first = t.update([seen("dog", box(20, 10))], 300)[0];
  const later = t.update([seen("dog", box(30, 10))], 600)[0];
  assert.equal(later.id, first.id);
  assert.equal(later.hits, 3);
});

test("a second object of the same kind gets its own identity", () => {
  const t = new Tracker(640 * 480);
  t.update([seen("dog", box(10, 10)), seen("dog", box(400, 300))], 0);
  const tracks = t.update([seen("dog", box(14, 10)), seen("dog", box(404, 300))], 300);
  assert.equal(new Set(tracks.map((x) => x.id)).size, 2);
});

test("something that jumps further than the gate allows is treated as new", () => {
  const t = new Tracker(640 * 480);
  t.update([seen("dog", box(10, 10))], 0);
  const first = t.update([seen("dog", box(12, 10))], 300)[0];
  // A box with no overlap at all cannot be the same object.
  const after = t.update([seen("dog", box(500, 400))], 600);
  assert.ok(!after.some((x) => x.id === first.id && x.box.x1 > 400));
});

test("labels are never confused with one another", () => {
  const t = new Tracker(640 * 480);
  t.update([seen("dog", box(10, 10))], 0);
  t.update([seen("dog", box(12, 10))], 300);
  const tracks = t.update([seen("cat", box(12, 10))], 600);
  assert.ok(!tracks.some((x) => x.label === "dog" && x.hits > 2));
});

test("a track survives a missed pass and is retired after too many", () => {
  const t = new Tracker(640 * 480);
  t.update([seen("dog", box(10, 10))], 0);
  t.update([seen("dog", box(12, 10))], 300);
  for (let i = 0; i <= MAX_MISSES; i++) t.update([], 600 + i * 300);
  assert.equal(t.all().length, 0, "retired once it stops being seen");
});

test("between observations a moving track is carried forward, not left behind", () => {
  const t = new Tracker(640 * 480);
  t.update([seen("dog", box(0, 0))], 0);
  t.update([seen("dog", box(30, 0))], 300);
  const [track] = t.update([seen("dog", box(60, 0))], 600);
  assert.ok(track.velocity.x > 0, "velocity points the way it is going");

  const now = predictBox(track, 900);
  assert.ok(centre(now).x > centre(track.box).x, "predicted ahead of the last sighting");
});

test("steady motion holds its speed instead of decaying to a standstill", () => {
  const t = new Tracker(640 * 480);
  // 30px every 300ms: 0.1 px/ms, held for two seconds.
  let track;
  for (let i = 0; i <= 7; i++) {
    [track] = t.update([seen("dog", box(i * 30, 0))], i * 300);
  }
  assert.ok(track, "a track survives the whole run");
  assert.ok(
    track.velocity.x > 0.05,
    `steady 0.1 px/ms should not decay; measured ${track.velocity.x.toFixed(4)}`,
  );
  assert.match(heading(track), /moving right/);
});

test("a still object is not given a heading", () => {
  const t = new Tracker(640 * 480);
  t.update([seen("dog", box(10, 10))], 0);
  const [track] = t.update([seen("dog", box(10, 10))], 300);
  assert.equal(heading(track), "holding still");
});

test("a box filling more of the frame reads as approaching", () => {
  const t = new Tracker(640 * 480);
  t.update([seen("dog", box(100, 100, 50, 50))], 0);
  t.update([seen("dog", box(100, 100, 60, 60))], 300);
  const [track] = t.update([seen("dog", box(100, 100, 120, 120))], 600);
  assert.equal(track.areaTrend, "growing");
});

test("the overlap gate is the thing separating identity from coincidence", () => {
  // Two boxes that overlap just under the gate must not be linked.
  const a = box(0, 0);
  const b = box(84, 0);
  assert.ok(iou(a, b) < IOU_GATE);
});

test("a track is forgotten on time, not on how fast the detector happens to run", () => {
  const t = new Tracker(640 * 480);
  t.update([seen("dog", box(10, 10))], 0);
  t.update([seen("dog", box(12, 10))], 300);
  // One slow pass, longer than the unseen window, with nothing in it.
  t.update([], 2200);
  assert.equal(t.all().length, 0);
});

test("the largest thing is reported first, being the likeliest subject", () => {
  const t = new Tracker(640 * 480);
  const small = seen("bird", box(0, 0, 20, 20));
  const large = seen("dog", box(200, 200, 300, 300));
  t.update([small, large], 0);
  const tracks = t.update([small, large], 300);
  assert.equal(tracks[0].label, "dog");
});

test("a growing box is described as approaching even when its centre barely moves", () => {
  const t = new Tracker(640 * 480);
  t.update([seen("dog", box(100, 100, 50, 50))], 0);
  t.update([seen("dog", box(99, 99, 70, 70))], 300);
  const [track] = t.update([seen("dog", box(98, 98, 140, 140))], 600);
  assert.match(heading(track), /coming closer/);
  assert.doesNotMatch(heading(track), /holding still, /);
});

test("a parked car becomes scenery while a crossing dog does not", () => {
  const t = new Tracker(640 * 480);
  // Same size, same label count, different behaviour: one still, one crossing.
  for (let i = 0; i <= 12; i++) {
    t.update(
      [
        seen("car", box(400, 300, 80, 80)),
        seen("dog", box(20 + i * 40, 200, 80, 80)),
      ],
      i * 300,
    );
  }
  const now = 12 * 300;
  const salient = t.salient(now);
  const scenery = t.scenery(now);
  assert.equal(salient.length, 1, "only the moving thing is worth reporting");
  assert.equal(salient[0].label, "dog");
  assert.deepEqual(scenery.map((s) => s.label), ["car"]);
});

test("something newly arrived is attended to before it has moved", () => {
  const t = new Tracker(640 * 480);
  for (let i = 0; i <= 12; i++) t.update([seen("car", box(400, 300))], i * 300);
  // A person appears late and stands still.
  const arrival = 13 * 300;
  t.update([seen("car", box(400, 300)), seen("person", box(100, 100))], arrival);
  t.update([seen("car", box(400, 300)), seen("person", box(100, 100))], arrival + 300);
  const salient = t.salient(arrival + 300);
  assert.ok(salient.some((s) => s.label === "person"), "a new arrival is not scenery yet");
});

test("the crowded frame is cut to a handful, not reported entire", () => {
  const t = new Tracker(640 * 480);
  // Eight parked things and two movers, as a wide shot actually looks.
  for (let i = 0; i <= 12; i++) {
    const seenNow = [];
    for (let k = 0; k < 8; k++) seenNow.push(seen("car", box(20 + k * 70, 380, 50, 50)));
    seenNow.push(seen("dog", box(20 + i * 40, 200, 60, 60)));
    seenNow.push(seen("person", box(600 - i * 38, 180, 60, 60)));
    t.update(seenNow, i * 300);
  }
  const salient = t.salient(12 * 300);
  assert.ok(salient.length <= 3, `reported ${salient.length}, should be a handful`);
  assert.deepEqual(
    salient.map((s) => s.label).sort(),
    ["dog", "person"],
    "the two movers, none of the eight parked",
  );
});

test("pace is measured against the frame, not in raw pixels", () => {
  const small = new Tracker(320 * 240);
  const large = new Tracker(1280 * 960);
  for (let i = 0; i <= 8; i++) {
    // Each crosses the same *fraction* of its own frame per step.
    small.update([seen("dog", box(i * 20, 50, 30, 30))], i * 300);
    large.update([seen("dog", box(i * 80, 200, 120, 120))], i * 300);
  }
  const a = small.salient(8 * 300)[0];
  const b = large.salient(8 * 300)[0];
  assert.ok(a && b, "both are salient");
  assert.ok(Math.abs(a.pace - b.pace) < 0.05, `pace ${a.pace.toFixed(3)} vs ${b.pace.toFixed(3)}`);
});

test("the overlap gate relaxes as the detector slows down", () => {
  assert.equal(gateFor(0), IOU_GATE, "no relaxation when the looks are simultaneous");
  assert.ok(gateFor(900) < IOU_GATE, "a slower detector demands less overlap");
  assert.ok(gateFor(5000) >= 0.05, "but never nothing at all");
});

test("a fast subject keeps its identity even when looks are far apart", () => {
  const t = new Tracker(640 * 480);
  // 900ms between passes, moving far enough that the boxes barely overlap.
  t.update([seen("dog", box(0, 200))], 0);
  const first = t.update([seen("dog", box(85, 200))], 900)[0];
  const later = t.update([seen("dog", box(170, 200))], 1800)[0];
  assert.ok(first && later, "the track survives both gaps");
  assert.equal(later.id, first.id, "and keeps one identity across them");
});

test("a panning camera does not turn the scenery into subjects", () => {
  const t = new Tracker(640 * 480);
  // Everything shifts together by 25px a step: that is the camera, not the world.
  for (let i = 0; i <= 12; i++) {
    t.update(
      [
        seen("car", box(300 - i * 25, 300, 70, 70)),
        seen("umbrella", box(420 - i * 25, 120, 70, 70)),
        seen("bench", box(520 - i * 25, 320, 70, 70)),
      ],
      i * 300,
    );
  }
  assert.equal(t.salient(12 * 300).length, 0, "a pan alone promotes nothing");
  assert.equal(t.scenery(12 * 300).length, 3, "all three are scenery");
});

test("against that pan, something moving differently still stands out", () => {
  const t = new Tracker(640 * 480);
  for (let i = 0; i <= 12; i++) {
    t.update(
      [
        seen("car", box(300 - i * 25, 300, 70, 70)),
        seen("umbrella", box(420 - i * 25, 120, 70, 70)),
        seen("bench", box(520 - i * 25, 320, 70, 70)),
        // Crossing the other way while the camera pans left.
        seen("dog", box(40 + i * 30, 220, 60, 60)),
      ],
      i * 300,
    );
  }
  const salient = t.salient(12 * 300);
  assert.deepEqual(salient.map((s) => s.label), ["dog"], "only the dog is doing anything");
});
