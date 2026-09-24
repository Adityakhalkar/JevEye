/**
 * Tracking quality, measured rather than eyeballed.
 *
 * Real MOT benchmarks (MOT17 and friends) are gigabytes and need licences, so
 * these are synthetic sequences with ground truth known by construction:
 * objects on fixed paths, with the failures that actually break trackers —
 * occlusion, detector jitter, and two things of the same kind crossing.
 *
 * The headline number is identity switches. A tracker that relabels a subject
 * halfway across the frame has failed at the one job detection cannot do.
 */
import { Tracker, iou } from "../src/lib/track.ts";

const FRAME = { w: 640, h: 480 };
const STEP_MS = 300;

const box = (x, y, w = 60, h = 60) => ({ x1: x, y1: y, x2: x + w, y2: y + h });

/** Deterministic jitter, so a run is reproducible. */
function jitter(seed) {
  let s = seed;
  return (amount) => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return ((s / 0x7fffffff) * 2 - 1) * amount;
  };
}

/** Each scenario yields, per frame, the true objects and what the detector saw. */
const SCENARIOS = {
  "one subject crossing the frame": () => {
    const frames = [];
    for (let i = 0; i < 20; i++) {
      const truth = [{ id: "A", label: "dog", box: box(20 + i * 28, 200) }];
      frames.push({ truth, seen: truth.map((t) => ({ ...t, score: 0.95 })) });
    }
    return frames;
  },

  "a subject the detector loses for three passes": () => {
    const frames = [];
    for (let i = 0; i < 20; i++) {
      const truth = [{ id: "A", label: "dog", box: box(20 + i * 28, 200) }];
      // Occluded in the middle: the object is there, the detector is not seeing it.
      const hidden = i >= 8 && i <= 10;
      frames.push({ truth, seen: hidden ? [] : truth.map((t) => ({ ...t, score: 0.9 })) });
    }
    return frames;
  },

  "noisy boxes around a steady subject": () => {
    const noise = jitter(7);
    const frames = [];
    for (let i = 0; i < 20; i++) {
      const truth = [{ id: "A", label: "dog", box: box(20 + i * 28, 200) }];
      const seen = truth.map((t) => ({
        ...t,
        score: 0.8,
        box: {
          x1: t.box.x1 + noise(9),
          y1: t.box.y1 + noise(9),
          x2: t.box.x2 + noise(9),
          y2: t.box.y2 + noise(9),
        },
      }));
      frames.push({ truth, seen });
    }
    return frames;
  },

  "two of the same kind crossing paths": () => {
    const frames = [];
    for (let i = 0; i < 20; i++) {
      const truth = [
        { id: "A", label: "person", box: box(20 + i * 28, 200) },
        { id: "B", label: "person", box: box(580 - i * 28, 200) },
      ];
      frames.push({ truth, seen: truth.map((t) => ({ ...t, score: 0.9 })) });
    }
    return frames;
  },
};

/** Which track best explains a ground-truth object this frame. */
function bestMatch(tracks, truth) {
  let best = null;
  let bestOverlap = 0.3;
  for (const t of tracks) {
    if (t.label !== truth.label) continue;
    const overlap = iou(t.box, truth.box);
    if (overlap > bestOverlap) {
      bestOverlap = overlap;
      best = t;
    }
  }
  return best;
}

function run(name, frames) {
  const tracker = new Tracker(FRAME.w * FRAME.h);
  const assigned = new Map(); // ground-truth id -> track id it was last matched to
  let switches = 0;
  let covered = 0;
  let total = 0;
  let ghosts = 0;

  frames.forEach((frame, i) => {
    const now = i * STEP_MS;
    tracker.update(
      frame.seen.map((s) => ({ label: s.label, score: s.score, box: s.box })),
      now,
    );
    // Judge against predicted positions: this is what a viewer sees.
    const tracks = tracker.predict(now);

    for (const truth of frame.truth) {
      total += 1;
      const match = bestMatch(tracks, truth);
      if (!match) continue;
      covered += 1;
      const previous = assigned.get(truth.id);
      if (previous !== undefined && previous !== match.id) switches += 1;
      assigned.set(truth.id, match.id);
    }

    // Tracks matching nothing real are inventions.
    const claimed = new Set(
      frame.truth.map((t) => bestMatch(tracks, t)?.id).filter((id) => id !== undefined),
    );
    ghosts += tracks.filter((t) => !claimed.has(t.id)).length;
  });

  return {
    name,
    switches,
    coverage: covered / total,
    ghostFrames: ghosts,
  };
}

/** What counts as a regression. Set from observed behaviour, not from hope. */
const THRESHOLDS = {
  "one subject crossing the frame": { switches: 0, coverage: 0.85, ghostFrames: 0 },
  "a subject the detector loses for three passes": { switches: 1, coverage: 0.6, ghostFrames: 2 },
  "noisy boxes around a steady subject": { switches: 0, coverage: 0.85, ghostFrames: 0 },
  "two of the same kind crossing paths": { switches: 2, coverage: 0.8, ghostFrames: 2 },
};

export function evaluateTracker() {
  const results = Object.entries(SCENARIOS).map(([name, build]) => run(name, build()));
  let failures = 0;

  console.log("\nTRACKING");
  for (const r of results) {
    const limit = THRESHOLDS[r.name];
    const ok =
      r.switches <= limit.switches &&
      r.coverage >= limit.coverage &&
      r.ghostFrames <= limit.ghostFrames;
    if (!ok) failures += 1;
    console.log(
      `  ${ok ? "pass" : "FAIL"}  ${r.name.padEnd(46)} ` +
        `id switches ${r.switches} (max ${limit.switches}), ` +
        `covered ${(r.coverage * 100).toFixed(0)}% (min ${limit.coverage * 100}%), ` +
        `phantom boxes ${r.ghostFrames} (max ${limit.ghostFrames})`,
    );
  }
  return failures;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  process.exit(evaluateTracker() > 0 ? 1 : 0);
}
