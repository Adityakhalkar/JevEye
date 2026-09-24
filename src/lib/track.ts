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
};

/** Overlap below this is not the same object, however alike the labels. */
export const IOU_GATE = 0.2;
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

  constructor(frameArea = 1) {
    this.frameArea = frameArea || 1;
  }

  /** The frame changed size, so shares of it have to be recomputed against that. */
  setFrameArea(value: number) {
    this.frameArea = value || 1;
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
      observations.forEach((o, j) => {
        if (o.label !== p.track.label) return;
        const overlap = iou(p.box, o.box);
        if (overlap >= IOU_GATE) pairs.push({ i, j, overlap });
      });
    });
    pairs.sort((a, b) => b.overlap - a.overlap);

    const takenTracks = new Set<number>();
    const takenObs = new Set<number>();
    for (const { i, j } of pairs) {
      if (takenTracks.has(i) || takenObs.has(j)) continue;
      takenTracks.add(i);
      takenObs.add(j);
      this.observe(predicted[i].track, predicted[i].box, observations[j], now);
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
      });
    });

    this.tracks = this.tracks.filter(
      (t) => t.misses <= MAX_MISSES && now - t.lastSeen <= MAX_UNSEEN_MS,
    );
    return this.confirmed();
  }

  private observe(track: Track, predictedBox: Box, observation: Observation, now: number) {
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
