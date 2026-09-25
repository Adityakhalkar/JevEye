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

  /**
   * A crowd the detector resolves differently each pass.
   *
   * Measured on real footage: about 27 boxes a pass, of which only ~30% overlap
   * anything from the pass before — and that share does not improve if weak
   * boxes are dropped (flat from a 0.5 floor to 0.95), nor once a whole-frame
   * shift and zoom are allowed for (24% to 33%). The detector simply finds a
   * different subset of a crowd every time it looks. Nothing can associate boxes
   * to objects that were never detected, so what matters here is that the few
   * subjects worth following stay followed, and that identity stops being
   * invented dozens of times a second.
   *
   * Three movers worth watching, among twenty stationary things each detected
   * about a third of the time. Only the movers are ground truth: tracking some
   * of the clutter is not an error, inventing hundreds of identities is.
   */
  "a crowd resolved differently each pass": () => {
    const dice = jitter(11);
    const frames = [];
    const clutter = [];
    for (let k = 0; k < 20; k++) {
      clutter.push({ id: `c${k}`, label: "car", box: box(30 + (k % 5) * 120, 40 + Math.floor(k / 5) * 60, 50, 50) });
    }
    for (let i = 0; i < 24; i++) {
      const truth = [
        { id: "A", label: "dog", box: box(20 + i * 24, 300) },
        { id: "B", label: "person", box: box(600 - i * 22, 340) },
        { id: "C", label: "person", box: box(100 + i * 18, 200) },
      ];
      const seen = [
        ...truth.filter(() => dice(1) > -0.8).map((t) => ({ ...t, score: 0.9 })),
        ...clutter.filter(() => dice(1) > 0.3).map((t) => ({ ...t, score: 0.8 })),
      ];
      frames.push({ truth, visible: [...truth, ...clutter], seen });
    }
    return frames;
  },
  /**
   * A subject that turns between sightings.
   *
   * Every other case here moves in a straight line at a steady speed, which is
   * precisely what a constant-velocity prediction assumes — so prediction scores
   * nearly perfectly on them and the lock looks pointless. That is an artefact of
   * the scenarios, not a fact about tracking. The moment a subject changes
   * direction the prediction keeps going the old way and the box leaves the
   * subject behind, which is the failure this was all built for: a dog doubling
   * back while the detector is still thinking about the last frame.
   */
  "a subject that turns between sightings": () => {
    const frames = [];
    let x = 300;
    let step = 26;
    for (let i = 0; i < 24; i++) {
      if (i % 6 === 0 && i > 0) step = -step; // doubles back every six frames
      x += step;
      const truth = [{ id: "A", label: "dog", box: box(x, 240, 60, 60) }];
      frames.push({
        truth,
        visible: truth,
        // The detector speaks once every three frames, so a turn is always taken
        // on the prediction's word alone.
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
  let fit = 0;
  let total = 0;
  let ghosts = 0;
  const born = new Set();
  // How often the set of things being watched changes, and who holds a seat.
  let attentionChanges = 0;
  let attended = [];
  const seatFrames = new Map();

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
      fit += iou(truth.box, match.box);
      const twinned = frame.truth.some((o, oi) => oi !== ti && iou(o.box, truth.box) > AMBIGUOUS);
      if (twinned) return; // nothing to conclude, and the last assignment stands
      const previous = assigned.get(truth.id);
      if (previous !== undefined && previous !== match.id) switches += 1;
      assigned.set(truth.id, match.id);
    });

    // Tracks matching nothing real are inventions.
    ghosts += tracks.filter((_, ki) => !usedTracks.has(ki)).length;
    tracks.forEach((t) => born.add(t.id));

    const watching = tracker.salient(now).map((t) => t.id);
    watching.forEach((id) => seatFrames.set(id, (seatFrames.get(id) ?? 0) + 1));
    if (watching.length !== attended.length || watching.some((id) => !attended.includes(id))) {
      attentionChanges += 1;
    }
    attended = watching;
  });

  return {
    name,
    switches,
    coverage: covered / total,
    // How well a drawn box actually sits on its subject. Coverage says a box was
    // there; this says it was in the right place, which is the lock's whole job.
    fit: covered ? fit / covered : 0,
    ghostFrames: ghosts,
    // One identity per real object is the ideal; more means the tracker keeps
    // rediscovering the same thing.
    births: born.size,
    attentionChanges,
    // How long a subject keeps attention once it has it, in frames.
    heldFor: seatFrames.size
      ? [...seatFrames.values()].reduce((a, b) => a + b, 0) / seatFrames.size
      : 0,
    // Everything actually in shot, not only what is scored: one identity per
    // object present is the ideal, and more than that is churn.
    objects: new Set(frames.flatMap((f) => (f.visible ?? f.truth).map((t) => t.id))).size,
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
  // Clutter may be tracked; identity may not be invented without limit. The
  // three movers are what must stay followed.
  // 23 things are in shot. Roughly one identity each is the goal; half again as
  // many means the tracker is rediscovering what it already had.
  // Prediction cannot do this one: the whole point is that the box has to be
  // placed by looking, because the assumption it would otherwise be placed by is
  // wrong the moment the subject turns.
  "a subject that turns between sightings": { switches: 0, coverage: 0.8, ghostFrames: 2, births: 1, fit: 0.6 },
  "a crowd resolved differently each pass": {
    switches: 2,
    coverage: 0.7,
    ghostFrames: 400,
    births: 26,
    // Three subjects are worth watching among twenty that are not. Attention
    // should settle on them, not be renegotiated every time the detector speaks.
    attentionChanges: 8,
  },
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
      r.births <= (limit.births ?? Infinity) &&
      r.attentionChanges <= (limit.attentionChanges ?? Infinity) &&
      r.fit >= (limit.fit ?? 0);
    if (!ok) failures += 1;
    console.log(
      `  ${ok ? "pass" : "FAIL"}  ${r.name.padEnd(46)} ` +
        `id switches ${r.switches} (max ${limit.switches}), ` +
        `covered ${(r.coverage * 100).toFixed(0)}% (min ${limit.coverage * 100}%), ` +
        `box sat on it ${(r.fit * 100).toFixed(0)}%${limit.fit ? ` (min ${limit.fit * 100}%)` : ""}, ` +
        `phantom boxes ${r.ghostFrames} (max ${limit.ghostFrames})` +
        (limit.births === undefined
          ? ""
          : `, identities ${r.births} for ${r.objects} object${r.objects === 1 ? "" : "s"} in view (max ${limit.births})`) +
        (limit.attentionChanges === undefined
          ? ""
          : `, attention changed ${r.attentionChanges}x (max ${limit.attentionChanges}), each subject held it ${r.heldFor.toFixed(1)} frames`),
    );
    const heldMore = (r.coverage - r.blind.coverage) * 100;
    const satBetter = (r.fit - r.blind.fit) * 100;
    const verdict =
      Math.abs(heldMore) < 0.5 && Math.abs(satBetter) < 0.5
        ? "the lock makes no difference here"
        : [
            Math.abs(heldMore) >= 0.5
              ? `${heldMore > 0 ? "+" : ""}${heldMore.toFixed(0)} points of coverage`
              : null,
            Math.abs(satBetter) >= 0.5
              ? `${satBetter > 0 ? "+" : ""}${satBetter.toFixed(0)} points of fit`
              : null,
          ]
            .filter(Boolean)
            .join(", ") + " from the lock";
    console.log(
      `        without the visual lock: covered ${(r.blind.coverage * 100).toFixed(0)}%, ` +
        `box sat on it ${(r.blind.fit * 100).toFixed(0)}%, id switches ${r.blind.switches}, ` +
        `identities ${r.blind.births} — ${verdict}`,
    );
  }
  return failures;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  process.exit(evaluateTracker() > 0 ? 1 : 0);
}
