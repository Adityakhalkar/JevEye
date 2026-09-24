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
import { frameOfPaint } from "../src/lib/scene.mock.ts";

const FRAME = { w: 640, h: 480 };
const STEP_MS = 300;

const box = (x, y, w = 60, h = 60) => ({ x1: x, y1: y, x2: x + w, y2: y + h });

/**
 * What a subject looks like, from its identity.
 *
 * The lock can only be measured against subjects that have an appearance, and
 * they have to differ from each other or the two-people scenarios would be
 * testing nothing. Three dark discs placed by a hash of the id: distinctive,
 * non-repeating, and nothing like a grating, which would match itself at a
 * shift and make a false lock look like a good one.
 */
function texture(id, u, v) {
  const hash = [...id].reduce((a, c) => (a * 131 + c.charCodeAt(0)) & 0xffff, 7);
  let value = 225;
  for (let k = 0; k < 3; k++) {
    const cx = 0.2 + (0.6 * ((hash >> (k * 5)) & 7)) / 7;
    const cy = 0.2 + (0.6 * ((hash >> (k * 5 + 2)) & 7)) / 7;
    if (Math.hypot(u - cx, v - cy) < 0.15) value -= 75;
  }
  return value;
}

/** The frame a viewer would see: whatever is actually in shot, on a plain field. */
function render(visible) {
  return frameOfPaint((x, y) => {
    for (const t of visible) {
      const w = t.box.x2 - t.box.x1;
      const h = t.box.y2 - t.box.y1;
      const u = (x - t.box.x1) / w;
      const v = (y - t.box.y1) / h;
      if (u < 0 || v < 0 || u >= 1 || v >= 1) continue;
      return texture(t.id, u, v);
    }
    return 90;
  }, FRAME.w, FRAME.h);
}

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
      // Still plainly in shot — this is the detector's failure, not an occlusion,
      // and the case where looking at the pixels should keep the box alive.
      frames.push({ truth, visible: truth, seen: hidden ? [] : truth.map((t) => ({ ...t, score: 0.9 })) });
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

  /**
   * The case appearance exists for: two things cross while the detector cannot
   * see them, and on reappearance they are on each other's old side. Overlap
   * has nothing to go on — the prediction for each lands where the other now
   * is — so this is where identity is won or lost.
   */
  "crossing behind an occlusion": () => crossingWithGap(),

  /**
   * One subject, approaching, while the detector speaks every third frame.
   *
   * This is the ordinary case on real footage and the one that churns identity:
   * a thing coming closer grows, so by the time the detector next finds it its
   * box is a different size from the one being carried forward. If association
   * demands too much overlap the sighting is taken for a new object, and what a
   * viewer sees is a subject that keeps being freshly discovered.
   */
  "coming closer while the detector lags": () => {
    const frames = [];
    for (let i = 0; i < 24; i++) {
      const size = 40 + i * 6;
      const truth = [{ id: "A", label: "dog", box: box(300 - size / 2, 240 - size / 2, size, size) }];
      frames.push({
        truth,
        visible: truth,
        seen: i % 3 === 0 ? truth.map((t) => ({ ...t, score: 0.9 })) : [],
      });
    }
    return frames;
  },

};

function crossingWithGap() {
  const frames = [];
  for (let i = 0; i < 20; i++) {
    const truth = [
      { id: "A", label: "person", box: box(20 + i * 28, 200) },
      { id: "B", label: "person", box: box(580 - i * 28, 200) },
    ];
    // Hidden exactly while they pass through one another.
    const hidden = i >= 9 && i <= 11;
    frames.push({
      truth,
      // Genuinely behind something: there is nothing to look at, so a lock must
      // not claim one. Identity has to survive the gap on prediction alone.
      visible: hidden ? [] : truth,
      seen: hidden ? [] : truth.map((t) => ({ ...t, score: 0.9 })),
    });
  }
  return frames;
}

/**
 * Assign ground truth to tracks one-to-one, best overlap first.
 *
 * Matching each object independently is wrong precisely where it matters: when
 * two things cross, both can claim the same track, and the scorer reports an
 * identity switch the tracker never made. MOT metrics use a one-to-one
 * assignment for this reason, and a scorer that invents failures is worse than
 * no scorer at all.
 */
function assign(tracks, truths, gate = 0.3) {
  const pairs = [];
  truths.forEach((truth, ti) => {
    tracks.forEach((track, ki) => {
      if (track.label !== truth.label) return;
      const overlap = iou(track.box, truth.box);
      if (overlap >= gate) pairs.push({ ti, ki, overlap });
    });
  });
  pairs.sort((a, b) => b.overlap - a.overlap);

  const matched = new Map();
  const usedTracks = new Set();
  for (const { ti, ki } of pairs) {
    if (matched.has(ti) || usedTracks.has(ki)) continue;
    matched.set(ti, tracks[ki]);
    usedTracks.add(ki);
  }
  return { matched, usedTracks };
}

/**
 * Overlap at which two ground-truth objects stop being tellable apart.
 *
 * Where two subjects occupy the same pixels there is no fact of the matter about
 * which box belongs to which: a one-to-one assignment picks one pairing, the next
 * frame picks the other, and the difference is recorded as two identity switches
 * the tracker never made. That is not leniency — the frames either side are still
 * scored, so a tracker that genuinely swaps two subjects is caught the moment
 * they separate, which is the only place the swap is visible.
 */
const AMBIGUOUS = 0.5;

function run(name, frames, { lock } = { lock: true }) {
  const tracker = new Tracker(FRAME.w * FRAME.h);
  const assigned = new Map(); // ground-truth id -> track id it was last matched to
  let switches = 0;
  let covered = 0;
  let total = 0;
  let ghosts = 0;
  const born = new Set();

  frames.forEach((frame, i) => {
    const now = i * STEP_MS;
    const sight = lock ? render(frame.visible ?? frame.truth) : undefined;
    tracker.update(
      frame.seen.map((s) => ({ label: s.label, score: s.score, box: s.box })),
      now,
      sight,
    );
    // Between detections the live page looks several times; once per step here
    // is the conservative version of the same thing.
    if (sight) tracker.look(sight, now);
    // Judge against predicted positions: this is what a viewer sees.
    const tracks = tracker.predict(now);

    const { matched, usedTracks } = assign(tracks, frame.truth);
    frame.truth.forEach((truth, ti) => {
      total += 1;
      const match = matched.get(ti);
      if (!match) return;
      covered += 1;
      const twinned = frame.truth.some((o, oi) => oi !== ti && iou(o.box, truth.box) > AMBIGUOUS);
      if (twinned) return; // nothing to conclude, and the last assignment stands
      const previous = assigned.get(truth.id);
      if (previous !== undefined && previous !== match.id) switches += 1;
      assigned.set(truth.id, match.id);
    });

    // Tracks matching nothing real are inventions.
    ghosts += tracks.filter((_, ki) => !usedTracks.has(ki)).length;
    tracks.forEach((t) => born.add(t.id));
  });

  return {
    name,
    switches,
    coverage: covered / total,
    ghostFrames: ghosts,
    // One identity per real object is the ideal; more means the tracker keeps
    // rediscovering the same thing.
    births: born.size,
    objects: new Set(frames.flatMap((f) => f.truth.map((t) => t.id))).size,
  };
}

/** What counts as a regression. Set from observed behaviour, not from hope. */
const THRESHOLDS = {
  "one subject crossing the frame": { switches: 0, coverage: 0.85, ghostFrames: 0 },
  "a subject the detector loses for three passes": { switches: 1, coverage: 0.6, ghostFrames: 2 },
  "noisy boxes around a steady subject": { switches: 0, coverage: 0.85, ghostFrames: 0 },
  "two of the same kind crossing paths": { switches: 2, coverage: 0.8, ghostFrames: 2 },
  // Appearance is the whole point here: a swap means the gate did not work.
  // Velocity carries each track through the gap on its own side, so identity
  // survives without appearance features.
  "crossing behind an occlusion": { switches: 0, coverage: 0.5, ghostFrames: 4 },
  "coming closer while the detector lags": { switches: 0, coverage: 0.7, ghostFrames: 2, births: 1 },
};

export function evaluateTracker() {
  const results = Object.entries(SCENARIOS).map(([name, build]) => ({
    ...run(name, build()),
    // The same sequence with the visual lock switched off, so its contribution
    // is a measured difference rather than a claim.
    blind: run(name, build(), { lock: false }),
  }));
  let failures = 0;

  console.log("\nTRACKING");
  for (const r of results) {
    const limit = THRESHOLDS[r.name];
    const ok =
      r.switches <= limit.switches &&
      r.coverage >= limit.coverage &&
      r.ghostFrames <= limit.ghostFrames &&
      r.births <= (limit.births ?? Infinity);
    if (!ok) failures += 1;
    console.log(
      `  ${ok ? "pass" : "FAIL"}  ${r.name.padEnd(46)} ` +
        `id switches ${r.switches} (max ${limit.switches}), ` +
        `covered ${(r.coverage * 100).toFixed(0)}% (min ${limit.coverage * 100}%), ` +
        `phantom boxes ${r.ghostFrames} (max ${limit.ghostFrames})` +
        (limit.births === undefined
          ? ""
          : `, identities ${r.births} for ${r.objects} object${r.objects === 1 ? "" : "s"} (max ${limit.births})`),
    );
    const gained = r.coverage - r.blind.coverage;
    console.log(
      `        without the visual lock: covered ${(r.blind.coverage * 100).toFixed(0)}%, ` +
        `id switches ${r.blind.switches}, identities ${r.blind.births}` +
        (Math.abs(gained) < 0.005
          ? " — the lock makes no difference here"
          : ` — the lock ${gained > 0 ? "adds" : "costs"} ${Math.abs(gained * 100).toFixed(0)} points of coverage`),
    );
  }
  return failures;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  process.exit(evaluateTracker() > 0 ? 1 : 0);
}
