/**
 * Multi-object tracking: turning independent detections into things that persist.
 *
 * A detector answers "what is in this frame" and nothing more. Run it twice and
 * you get two unrelated lists, which is why boxes jump: frame to frame there is
 * no claim that this dog is the same dog. Tracking adds that claim, and with it
 * the only questions worth asking of a moving scene — how long has this been
 * here, which way is it going, is it getting closer.
 *
 * The method is the standard one (SORT): associate by overlap, carry a velocity,
 * predict between observations, and retire what stops being seen. No Kalman
 * filter — a constant-velocity estimate with exponential smoothing behaves
 * almost identically at these frame rates and is far easier to reason about.
 */

export type Box = { x1: number; y1: number; x2: number; y2: number };

export type Observation = { label: string; score: number; box: Box };

export type Track = {
  id: number;
  label: string;
  /** Where it was last observed, smoothed against its previous position. */
  box: Box;
  /** Centre motion in pixels per millisecond. */
  velocity: { x: number; y: number };
  score: number;
  /** How many times it has been observed; a track is provisional until MIN_HITS. */
  hits: number;
  /** Consecutive detection passes that did not find it. */
  misses: number;
  firstSeen: number;
  lastSeen: number;
  /** Box area as a share of the frame, and whether that is growing. */
  area: number;
  areaTrend: "growing" | "shrinking" | "steady";
  /**
   * Distance the centre has travelled *relative to the scene*, in pixels.
   *
   * Absolute motion is the wrong measure the moment a camera pans: everything
   * in shot moves together, and parked cars become subjects. What distinguishes
   * a dog crossing a field is that it moves differently from everything else.
   */
  travelled: number;
  /** Share of the frame crossed per second, averaged over the track's life. */
  pace: number;
  /**
   * How much this deserves attention, against everything else in view.
   *
   * A detector finds every car, tent and umbrella in a wide shot, and reporting
   * them all is not understanding the scene — it is a list. What separates the
   * dog and its handler from the parked cars behind them is that they move.
   */
  salience: number;
  /** True for what has sat still long enough to be scenery. */
  background: boolean;
};

/** Overlap below this is not the same object, however alike the labels. */
export const IOU_GATE = 0.2;
/**
 * The gate relaxes as time passes between looks.
 *
 * Overlap is only evidence of identity when the two observations are close in
 * time. A track first seen a moment ago has no velocity to predict with, so
 * after a long gap the honest expectation is that it has moved some distance
 * and overlaps its old box far less. Holding a fixed gate across any interval
 * means a slower detector silently stops tracking anything.
 */
const GATE_HALVES_AFTER_MS = 500;
const GATE_FLOOR = 0.05;

export function gateFor(elapsedMs: number): number {
  const relaxed = IOU_GATE / (1 + Math.max(0, elapsedMs) / GATE_HALVES_AFTER_MS);
  return Math.max(GATE_FLOOR, relaxed);
}
/** Observations before a track is shown, so one bad frame invents nothing. */
export const MIN_HITS = 2;
/** Missed passes before a track is retired. */
export const MAX_MISSES = 3;
/**
 * How long something may go unseen before it is forgotten.
 *
 * Counting passes alone ties a track's lifetime to how fast the detector
 * happens to be running. In time, a brief occlusion — a dog behind a jump —
 * should not cost an identity, while something genuinely gone should not linger.
 */
export const MAX_UNSEEN_MS = 1500;
/** How much a new observation moves the box: 1 is jumpy, 0 never catches up. */
const SMOOTHING = 0.65;
/** How much a new velocity estimate moves the old one. */
const VELOCITY_SMOOTHING = 0.5;
/** Area changes below this are noise, not approach. */
const AREA_EPSILON = 0.15;
/** Crossing less of the frame per second than this is standing still. */
export const STILL_PACE = 0.012;
/** How long something must sit still before it counts as scenery. */
export const SCENERY_AFTER_MS = 2500;
/** Newly arrived things are worth attention even before they have moved. */
const NOVELTY_MS = 2000;

export function area(box: Box): number {
  return Math.max(0, box.x2 - box.x1) * Math.max(0, box.y2 - box.y1);
}

export function centre(box: Box): { x: number; y: number } {
  return { x: (box.x1 + box.x2) / 2, y: (box.y1 + box.y2) / 2 };
}

/** Intersection over union: how much two boxes agree about where something is. */
export function iou(a: Box, b: Box): number {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  const overlap = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = area(a) + area(b) - overlap;
  return union <= 0 ? 0 : overlap / union;
}

const lerp = (from: number, to: number, t: number) => from + (to - from) * t;

/**
 * How many things must be in view before their common drift can be called the
 * camera's.
 *
 * With two tracks there is no majority: the median lands on whichever moved,
 * and a lone subject gets mistaken for the scene it is crossing. Below this the
 * safe assumption is that the camera is still and the motion is real.
 */
const FLOW_NEEDS = 3;

/** Component-wise median of a set of shifts; zero when there are too few to trust. */
function median2(shifts: Array<{ x: number; y: number }>): { x: number; y: number } {
  if (shifts.length < FLOW_NEEDS) return { x: 0, y: 0 };
  const pick = (values: number[]) => {
    const sorted = [...values].sort((a, b) => a - b);
    return sorted[Math.floor(sorted.length / 2)];
  };
  return { x: pick(shifts.map((s) => s.x)), y: pick(shifts.map((s) => s.y)) };
}

function blend(from: Box, to: Box, t: number): Box {
  return {
    x1: lerp(from.x1, to.x1, t),
    y1: lerp(from.y1, to.y1, t),
    x2: lerp(from.x2, to.x2, t),
    y2: lerp(from.y2, to.y2, t),
  };
}

function shift(box: Box, dx: number, dy: number): Box {
  return { x1: box.x1 + dx, y1: box.y1 + dy, x2: box.x2 + dx, y2: box.y2 + dy };
}

export class Tracker {
  private tracks: Track[] = [];
  private nextId = 1;
  private frameArea: number;
  private frameSpan: number;

  constructor(frameArea = 1) {
    this.frameArea = frameArea || 1;
    this.frameSpan = Math.sqrt(this.frameArea) || 1;
  }

  /** The frame changed size, so shares of it have to be recomputed against that. */
  setFrameArea(value: number) {
    this.frameArea = value || 1;
    this.frameSpan = Math.sqrt(this.frameArea) || 1;
  }

  /**
   * Fold a detection pass into the existing tracks.
   *
   * Association is greedy on overlap rather than optimal (Hungarian): with a
   * handful of boxes the two agree almost always, and greedy is far easier to
   * follow when a result looks wrong.
   */
  update(observations: Observation[], now: number): Track[] {
    const predicted = this.tracks.map((t) => ({ track: t, box: predictBox(t, now) }));
    const pairs: Array<{ i: number; j: number; overlap: number }> = [];
    predicted.forEach((p, i) => {
      // How much overlap to demand depends on how long it has been since this
      // track was last seen; see `gateFor`.
      const gate = gateFor(now - p.track.lastSeen);
      observations.forEach((o, j) => {
        if (o.label !== p.track.label) return;
        const overlap = iou(p.box, o.box);
        if (overlap >= gate) pairs.push({ i, j, overlap });
      });
    });
    pairs.sort((a, b) => b.overlap - a.overlap);

    const takenTracks = new Set<number>();
    const takenObs = new Set<number>();
    const matched: Array<{ track: Track; box: Box; observation: Observation; shift: { x: number; y: number } }> = [];
    for (const { i, j } of pairs) {
      if (takenTracks.has(i) || takenObs.has(j)) continue;
      takenTracks.add(i);
      takenObs.add(j);
      const from = centre(this.tracks[i] === predicted[i].track ? predicted[i].track.box : predicted[i].box);
      const to = centre(observations[j].box);
      matched.push({
        track: predicted[i].track,
        box: predicted[i].box,
        observation: observations[j],
        shift: { x: to.x - from.x, y: to.y - from.y },
      });
    }

    /**
     * The scene's own drift, taken as the median of everything that moved.
     *
     * A median rather than a mean: when a camera pans, most of what is in shot
     * is scenery carried along by it, and the few things genuinely moving
     * should not drag the estimate they are being measured against.
     */
    const flow = median2(matched.map((m) => m.shift));
    for (const m of matched) {
      this.observe(m.track, m.box, m.observation, now, flow);
    }

    predicted.forEach((p, i) => {
      if (takenTracks.has(i)) return;
      p.track.misses += 1;
    });

    observations.forEach((o, j) => {
      if (takenObs.has(j)) return;
      this.tracks.push({
        id: this.nextId++,
        label: o.label,
        box: o.box,
        velocity: { x: 0, y: 0 },
        score: o.score,
        hits: 1,
        misses: 0,
        firstSeen: now,
        lastSeen: now,
        area: area(o.box) / this.frameArea,
        areaTrend: "steady",
        travelled: 0,
        pace: 0,
        salience: 0,
        background: false,
      });
    });

    this.tracks = this.tracks.filter(
      (t) => t.misses <= MAX_MISSES && now - t.lastSeen <= MAX_UNSEEN_MS,
    );
    return this.confirmed();
  }

  private observe(
    track: Track,
    predictedBox: Box,
    observation: Observation,
    now: number,
    flow: { x: number; y: number } = { x: 0, y: 0 },
  ) {
    const dt = Math.max(now - track.lastSeen, 1);
    /**
     * Measured against the last *observation*, not the prediction.
     *
     * Against the prediction this measures the correction rather than the
     * motion, so a track moving at a steady speed reports a shrinking velocity
     * and is eventually described as holding still while it crosses the frame.
     */
    const before = centre(track.box);
    const after = centre(observation.box);
    const measured = { x: (after.x - before.x) / dt, y: (after.y - before.y) / dt };

    track.velocity = {
      x: lerp(track.velocity.x, measured.x, VELOCITY_SMOOTHING),
      y: lerp(track.velocity.y, measured.y, VELOCITY_SMOOTHING),
    };
    const previousArea = track.area;
    track.box = blend(predictedBox, observation.box, SMOOTHING);
    track.score = observation.score;
    track.hits += 1;
    track.misses = 0;
    track.lastSeen = now;
    track.area = area(track.box) / this.frameArea;

    const change = previousArea > 0 ? (track.area - previousArea) / previousArea : 0;
    track.areaTrend =
      change > AREA_EPSILON ? "growing" : change < -AREA_EPSILON ? "shrinking" : "steady";

    // Distance is accumulated rather than sampled: something that drifts
    // steadily and something that jitters in place look alike frame to frame,
    // and only the total separates them.
    // Measured against the scene's drift, so a pan does not promote scenery.
    const moved = Math.hypot(after.x - before.x - flow.x, after.y - before.y - flow.y);
    track.travelled += moved;
    this.rate(track, now);
  }

  /**
   * Motion as a share of the frame per second, and what follows from it.
   *
   * Pace rather than raw pixels, so the judgement holds whatever the video's
   * size. Size counts too, but less: a lorry parked across the shot is scenery,
   * while a small thing crossing it is the story.
   */
  private rate(track: Track, now: number) {
    const seconds = Math.max((now - track.firstSeen) / 1000, 0.3);
    track.pace = track.travelled / this.frameSpan / seconds;
    const fresh = now - track.firstSeen < NOVELTY_MS;
    track.background =
      !fresh && track.pace < STILL_PACE && now - track.firstSeen > SCENERY_AFTER_MS;
    track.salience = track.pace * 3 + Math.sqrt(track.area) * 0.5 + (fresh ? 0.15 : 0);
  }

  /**
   * The few things worth reporting, most deserving first.
   *
   * Everything the detector found is still tracked — a question about the
   * background can still be answered — but scenery is not what the view is
   * about, and a list of eleven parked cars is not an understanding of it.
   */
  salient(now: number, limit = 3): Track[] {
    return this.predict(now)
      .filter((t) => !t.background)
      .sort((a, b) => b.salience - a.salience)
      .slice(0, limit);
  }

  /** Everything currently held, scenery included. */
  scenery(now: number): Track[] {
    return this.predict(now).filter((t) => t.background);
  }

  /**
   * Where everything is now, between detection passes.
   *
   * This is what makes motion look continuous: the detector speaks roughly once
   * a second, and in between each track is carried forward along its own
   * velocity rather than sitting still and then teleporting.
   */
  predict(now: number): Track[] {
    return this.confirmed().map((t) => ({ ...t, box: predictBox(t, now) }));
  }

  private confirmed(): Track[] {
    return this.tracks
      .filter((t) => t.hits >= MIN_HITS && t.misses === 0)
      .sort((a, b) => b.area - a.area);
  }

  /** Every live track, including ones currently unobserved. */
  all(): Track[] {
    return this.tracks.filter((t) => t.hits >= MIN_HITS);
  }

  reset() {
    this.tracks = [];
    this.nextId = 1;
  }
}

/** A track's box carried forward to `now` along its velocity. */
export function predictBox(track: Track, now: number): Box {
  const dt = Math.max(0, now - track.lastSeen);
  return shift(track.box, track.velocity.x * dt, track.velocity.y * dt);
}

/**
 * What a track is doing, in one phrase.
 *
 * Depth and direction are reported together because saying both separately
 * produces nonsense like "holding still, moving away": a box that is growing
 * is moving, whatever its centre is doing.
 */
export function heading(track: Track): string {
  const speed = Math.hypot(track.velocity.x, track.velocity.y) * 1000;
  const depth =
    track.areaTrend === "growing"
      ? "coming closer"
      : track.areaTrend === "shrinking"
        ? "moving away"
        : null;

  if (speed < 15) return depth ?? "holding still";

  const horizontal = Math.abs(track.velocity.x) > Math.abs(track.velocity.y);
  const direction = horizontal
    ? track.velocity.x > 0
      ? "moving right"
      : "moving left"
    : track.velocity.y > 0
      ? "moving down"
      : "moving up";
  return depth ? `${direction} and ${depth}` : direction;
}
