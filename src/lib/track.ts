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

import { enough, freshen, print, relock, verify, type Frame, type Print } from "./lock.ts";

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
  /**
   * When the detector last found it, as distinct from when it was last seen.
   *
   * These part company once looking keeps a track alive on its own, and the
   * difference matters for association. A lock says the thing is still there; it
   * says nothing about the shape of the box, which it holds at a fixed size while
   * the subject turns or approaches. So how much overlap to demand of the next
   * sighting has to relax with time since the *detector* last agreed about the
   * geometry. Timing it from the lock instead held every track at the strictest
   * gate and only 39% of detections ever found their own track again.
   */
  lastDetected: number;
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
  /**
   * What this looked like when the detector last confirmed it.
   *
   * Taken at identification and never refreshed from the lock's own output: a
   * template rebuilt from where the tracker *thinks* the subject is will follow
   * its own mistakes, and a box that drifts a pixel a frame ends up on the
   * background with nothing to notice it went. Anchored to the last real
   * sighting, error cannot compound.
   */
  template: Print | null;
  /** How well the frame still matches that print: 1 at a sighting, 0 when lost. */
  lock: number;
  /**
   * Whether appearance can settle whether this is still there.
   *
   * A subject of seventeen pixels has no print worth matching, and treating the
   * failure as absence is what made attention flicker: the thing was plainly in
   * shot, the detector merely had not mentioned it this pass, and the lock could
   * not speak for it either. When it cannot, silence has to be read as silence.
   */
  lockable: boolean;
};

/** Overlap below this is not the same object, however alike the labels. */
export const IOU_GATE = 0.2;

/**
 * Correlation below this is not worth moving a box for.
 *
 * Set from measurement rather than taste: on prints of the same subject moved
 * across a frame the match holds around 0.8, while a contrast-inverted impostor
 * peaks near 0.64 and noise near 0.24. Half sits below anything real and above
 * anything measured that was not the subject, and a lock that falls short
 * simply leaves the box to its predicted position rather than guessing.
 */
export const LOCK_KEEP = 0.5;

/**
 * How much time must have passed for a lock to say anything about speed.
 *
 * A displacement divided by a millisecond is not a velocity, it is a rounding
 * error with a large number attached. Looking can land immediately after a
 * detection, and taking that as motion sent boxes flying off the frame at
 * hundreds of pixels a second. Below this the lock still places the box; it
 * just does not pretend to have measured how fast it got there.
 */
export const LOOK_MIN_MS = 30;

/**
 * Overlap at which two locks are claiming the same thing.
 *
 * When one subject passes in front of another the one behind has nothing left to
 * match, and the best patch in its neighbourhood is the subject in front — so it
 * locks onto that and the two identities swap. Two things cannot be in the same
 * place, which is the same one-to-one reasoning the detector association already
 * uses, so the weaker claim is dropped and that track falls back to prediction —
 * which is what carries identity through a crossing in the first place.
 */
export const LOCK_CLAIM = 0.5;

/**
 * How sure a look must be before it is allowed to update what a thing looks like.
 *
 * Comfortably above the threshold for moving a box: a match good enough to act on
 * is not necessarily good enough to learn from, and anything learned from a poor
 * match is how a print walks off its subject.
 */
export const LOCK_TRUST = 0.7;

/** How much of a trusted look is eased into the print each time. */
export const FRESHEN_RATE = 0.12;

/**
 * How long something is taken on trust after a miss, at least.
 *
 * Long enough to ride out a pass of a detector that finds a different third of a
 * crowd each time, short enough that what has actually left is not outlined for
 * long. Retirement still applies underneath.
 */
export const GRACE_MS = 2000;

/**
 * Patience, counted in the detector's own passes rather than in milliseconds.
 *
 * Every timeout here was set when the detector was assumed to speak often. It
 * does not: on real footage a pass takes some 1.3 seconds, which made the 1.5
 * second retirement barely one missed pass, and tracks were being forgotten
 * faster than the grace meant to protect them could apply — twenty-seven of the
 * forty-eight subjects that lost attention were simply retired out from under it.
 * How long to wait is a question about the detector's rhythm, so it is measured
 * from the detector's rhythm, and a slower machine is patient in proportion.
 */
const UNSEEN_PASSES = 2.5;
const GRACE_PASSES = 1.5;

/**
 * However slow the detector, this is as long as anything is taken on trust.
 *
 * Patience that scales without limit would let a machine having a bad minute
 * leave boxes on things that walked off ten seconds ago. Waiting for the
 * detector is reasonable; waiting indefinitely is not.
 */
const MAX_PATIENCE_MS = 4000;

/**
 * How quickly the cadence estimate follows a change, deliberately slowly.
 *
 * One pass that takes a while is noise, not a new rhythm, and letting a single
 * slow pass stretch the estimate would keep a track alive on the strength of an
 * anomaly. Slow enough to ignore a spike, quick enough to settle within a few
 * passes of a genuinely different machine.
 */
const CADENCE_SMOOTHING = 0.15;

/**
 * How big a box to cut around a point nothing was detected at, as a share of the
 * frame's span. Big enough to hold a subject, small enough to be mostly subject.
 */
const POINTED_AT = 0.16;

/** How many things are worth holding a box on, when nobody says otherwise. */
export const ATTEND_LIMIT = 3;

/**
 * How much more salient a newcomer must be to take attention from what has it.
 *
 * Attention had no memory: every frame ranked everything afresh, so with a
 * detector finding some eighteen new things a pass — half of them in places
 * nothing was being tracked — whatever had just arrived kept displacing whatever
 * was being followed. A newly created track has no measured pace yet and scores
 * on its size and a novelty bonus alone, which is how a parked lorry unseats a
 * running dog. Ties now go to whoever is already being watched.
 */
export const ATTENTION_MARGIN = 1.3;

/**
 * How long something must have been watched before it can unseat anything.
 *
 * Pace is distance over the track's lifetime, so at one sighting old it is
 * either zero or nonsense. This is the interval over which a newcomer has to
 * show what it is actually doing. It may still fill a seat nobody wants
 * immediately — the wait is only to take one from something else.
 */
export const AUDITION_MS = 1200;
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

/** Whether a point falls inside a region. */
function within(point: { x: number; y: number }, region: Box): boolean {
  return (
    point.x >= region.x1 && point.x <= region.x2 && point.y >= region.y1 && point.y <= region.y2
  );
}

function shift(box: Box, dx: number, dy: number): Box {
  return { x1: box.x1 + dx, y1: box.y1 + dy, x2: box.x2 + dx, y2: box.y2 + dy };
}

export class Tracker {
  private tracks: Track[] = [];
  private nextId = 1;
  /**
   * The one thing someone has asked for, which outranks every judgement here.
   *
   * Salience is a guess at what matters, and a good one, but a person pointing at
   * something is not a guess. A pinned track cannot be unseated, cannot be
   * demoted to scenery, and is always looked for by name.
   */
  private pinned: number | null = null;
  /** Which tracks currently hold attention, and when that was last settled. */
  private attending: number[] = [];
  private attendedAt = -1;
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
  /**
   * @param covered Which part of the frame this pass actually examined. A close
   * look at one region says nothing about the rest, and counting everything
   * outside it as missed would retire half the scene every time the detector
   * looked somewhere in detail.
   */
  update(observations: Observation[], now: number, frame?: Frame, covered?: Box): Track[] {
    if (this.lastUpdate >= 0) {
      const gap = now - this.lastUpdate;
      this.cadence = this.cadence ? lerp(this.cadence, gap, CADENCE_SMOOTHING) : gap;
    }
    this.lastUpdate = now;
    const predicted = this.tracks.map((t) => ({ track: t, box: predictBox(t, now) }));
    const pairs: Array<{ i: number; j: number; overlap: number }> = [];
    predicted.forEach((p, i) => {
      // How much overlap to demand depends on how long it has been since this
      // track was last seen; see `gateFor`.
      const gate = gateFor(now - p.track.lastDetected);
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
      this.observe(m.track, m.box, m.observation, now, flow, frame);
    }

    predicted.forEach((p, i) => {
      if (takenTracks.has(i)) return;
      if (covered && !within(centre(p.box), covered)) return; // nobody looked here
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
        lastDetected: now,
        area: area(o.box) / this.frameArea,
        areaTrend: "steady",
        travelled: 0,
        pace: 0,
        salience: 0,
        background: false,
        template: null,
        // No evidence yet that this is still in shot, and none is claimed: until
        // something has actually looked, only the detector's word keeps a box.
        lock: 0,
        lockable: false,
      });
      this.remember(this.tracks[this.tracks.length - 1], frame);
    });

    this.tracks = this.tracks.filter(
      (t) =>
        (t.id === this.pinned && t.lock >= LOCK_KEEP) ||
        (t.misses <= MAX_MISSES && now - t.lastSeen <= this.patience()),
    );
    return this.confirmed();
  }

  private observe(
    track: Track,
    predictedBox: Box,
    observation: Observation,
    now: number,
    flow: { x: number; y: number } = { x: 0, y: 0 },
    frame?: Frame,
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
    track.lastDetected = now;
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
    this.remember(track, frame);
  }

  /**
   * Having seen something, whoever saw it.
   *
   * Retirement exists to forget what has left, and a thing whose own appearance
   * is still where it should be has not left. The detector's silence is not
   * evidence of absence when it has been measured to mention only a third of
   * what is in front of it from one pass to the next. `hits` is left alone: that
   * counts identifications, which is the detector's word and not the lock's.
   */
  private sighted(track: Track, now: number) {
    track.lastSeen = now;
    track.misses = 0;
  }

  /**
   * Keep a print of what the detector just confirmed.
   *
   * A refused print — too small a box, or a patch so flat it would match
   * anywhere — leaves the previous one in place rather than clearing it: one
   * awkward crop is not a reason to throw away a working template.
   */
  private remember(track: Track, frame?: Frame) {
    if (!frame) return;
    track.template = print(frame, track.box) ?? track.template;
    if (track.template) track.lock = 1;
  }

  /**
   * Find every track again in the current frame, by its own appearance.
   *
   * This is the difference between a box that follows a subject and one that
   * follows an assumption. Between detector passes a velocity says where a
   * thing *ought* to be, which is wrong the instant it turns; the print says
   * where it *is*. Cheap enough — well under a millisecond each — to run on
   * every displayed frame.
   *
   * A successful lock counts as having seen the track, so something the
   * detector momentarily loses but which is plainly still in shot keeps its box
   * instead of blinking out. The detector's own miss count is untouched, so
   * MAX_MISSES still retires anything it has genuinely stopped finding.
   */
  look(frame: Frame, now: number, limit = ATTEND_LIMIT): void {
    /*
     * Only the few things actually being shown.
     *
     * Matching every parked car and umbrella in a wide shot is what took this
     * from one millisecond a frame to thirty, and none of those boxes is drawn:
     * the view attends to a handful, and the rest is scenery whose position
     * nobody is looking at. Ordered by salience so the ones on screen are the
     * ones kept, and the boxes that matter hold whatever else is in shot.
     */
    const holding = this.tracks.filter((t) => t.template);
    const searched = new Set(
      holding
        .filter((t) => !t.background)
        .sort((a, b) => {
          if (a.id === this.pinned) return -1;
          if (b.id === this.pinned) return 1;
          return b.salience - a.salience;
        })
        .slice(0, limit),
    );

    /*
     * Everything else is only asked whether it is still there.
     *
     * A detector that finds a different subset of a crowd every pass was retiring
     * tracks it had merely failed to mention, and the next sighting arrived as a
     * brand new object: thirty-five identities for three subjects. Confirming a
     * print in place costs a thirtieth of a search, so everything in view can keep
     * its identity through a dropout while only what is on screen pays to be
     * located precisely.
     */
    for (const track of holding) {
      if (searched.has(track) || !track.template) continue;
      track.lockable = enough(frame, track.box);
      if (!track.lockable) continue;
      const still = verify(frame, track.template, predictBox(track, now));
      track.lock = Math.max(still, 0);
      if (still >= LOCK_KEEP) this.sighted(track, now);
    }

    const claims: Array<{ track: Track; found: NonNullable<ReturnType<typeof relock>> }> = [];
    for (const track of searched) {
      if (!track.template) continue;
      track.lockable = enough(frame, track.box);
      if (!track.lockable) continue;
      const found = relock(frame, track.template, predictBox(track, now));
      track.lock = found ? Math.max(found.score, 0) : 0;
      if (found && found.score >= LOCK_KEEP) claims.push({ track, found });
    }

    // Strongest claim first, so a subject in plain view keeps what is its own and
    // the one that has lost sight of itself is the one asked to let go.
    claims.sort((a, b) => b.found.score - a.found.score);
    const taken: Box[] = [];
    for (const { track, found } of claims) {
      if (taken.some((box) => iou(box, found.rect) > LOCK_CLAIM)) {
        track.lock = 0;
        continue;
      }
      taken.push(found.rect);

      const elapsed = now - track.lastSeen;
      if (elapsed >= LOOK_MIN_MS) {
        const before = centre(track.box);
        const after = centre(found.rect);
        const measured = { x: (after.x - before.x) / elapsed, y: (after.y - before.y) / elapsed };
        track.velocity = {
          x: lerp(track.velocity.x, measured.x, VELOCITY_SMOOTHING),
          y: lerp(track.velocity.y, measured.y, VELOCITY_SMOOTHING),
        };
      }
      track.box = found.rect;
      this.sighted(track, now);
      const template = track.template;
      if (template && found.score >= LOCK_TRUST) {
        const fresh = print(frame, found.rect);
        if (fresh) track.template = freshen(template, fresh, FRESHEN_RATE);
      }
      // `travelled` is deliberately left alone. It is sampled by the detector at
      // its own rhythm, and salience is calibrated against that; folding in ten
      // sub-pixel corrections a second would let jitter accumulate into pace and
      // promote parked cars to subjects.
    }
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
      track.id !== this.pinned &&
      !fresh &&
      track.pace < STILL_PACE &&
      now - track.firstSeen > SCENERY_AFTER_MS;
    track.salience = track.pace * 3 + Math.sqrt(track.area) * 0.5 + (fresh ? 0.15 : 0);
  }

  /**
   * The few things worth reporting, most deserving first.
   *
   * Everything the detector found is still tracked — a question about the
   * background can still be answered — but scenery is not what the view is
   * about, and a list of eleven parked cars is not an understanding of it.
   */
  salient(now: number, limit = ATTEND_LIMIT): Track[] {
    const live = this.predict(now).filter((t) => !t.background);
    if (now !== this.attendedAt) {
      this.attendedAt = now;
      this.reconsider(live, now, limit);
    }
    const byId = new Map(live.map((t) => [t.id, t]));
    return this.attending
      .map((id) => byId.get(id))
      .filter((t): t is Track => t !== undefined)
      .sort((a, b) => b.salience - a.salience);
  }

  /**
   * Who is being watched, changed only for good reason.
   *
   * Nothing is watched until it has been watched, which sounds circular and is
   * the point: a seat is only given to something whose pace has had time to mean
   * something. Filling free seats immediately was the larger half of the churn —
   * seats come free constantly as subjects settle into scenery or leave, and
   * each one was handed to whatever fresh box happened to be biggest, which in a
   * crowded street is a different umbrella every time. Taking an *occupied* seat
   * needs more than patience: a clear margin over the weakest sitting subject,
   * so attention does not swap on a hair's difference.
   *
   * The honest cost is that a subject is not outlined for its first second, and
   * that a seat sometimes sits empty. Showing two boxes is better than filling
   * the third with whatever was nearest.
   */
  private reconsider(live: Track[], now: number, limit: number) {
    const byId = new Map(live.map((t) => [t.id, t]));
    const seated = this.attending.filter((id) => byId.has(id));
    if (this.pinned !== null && byId.has(this.pinned) && !seated.includes(this.pinned)) {
      seated.unshift(this.pinned);
    }
    const queue = live
      .filter((t) => !seated.includes(t.id))
      .sort((a, b) => b.salience - a.salience);

    /*
     * Proven subjects first, then novelty if seats are still going spare.
     *
     * An audition cannot be an absolute requirement or nothing would ever be
     * watched in the first second of a video, and someone walking into an empty
     * room — interesting precisely because they are new — would go unmarked. So
     * the rule is an order of preference: anything whose pace has been measured
     * is preferred, and only if seats remain does something newly arrived get
     * one. In a quiet scene it is the only candidate and is attended at once; in
     * a crowd there is always something proven, and novelty waits its turn.
     */
    const auditioned = queue.filter((t) => now - t.firstSeen >= AUDITION_MS);
    const newcomers = queue.filter((t) => now - t.firstSeen < AUDITION_MS);
    while (seated.length < limit && auditioned.length) seated.push(auditioned.shift()!.id);
    while (seated.length < limit && newcomers.length) seated.push(newcomers.shift()!.id);

    for (const challenger of auditioned) {
      let weakest = Infinity;
      let seat = -1;
      seated.forEach((id, i) => {
        if (id === this.pinned) return; // nobody takes the seat that was asked for
        const salience = byId.get(id)!.salience;
        if (salience < weakest) {
          weakest = salience;
          seat = i;
        }
      });
      if (seat < 0 || challenger.salience <= weakest * ATTENTION_MARGIN) continue;
      seated[seat] = challenger.id;
    }
    this.attending = seated.slice(0, limit);
  }

  /**
   * Follow whatever is at this point, detected or not.
   *
   * Preferring the smallest box containing the point, because in a crowd the
   * large ones are the background someone is pointing past. When nothing has been
   * detected there at all, a track is minted from the pixels themselves — the
   * detector's vocabulary is eighty nouns and the world is larger than that, and
   * following something it has no word for is exactly what appearance is good at.
   *
   * Returns the track now being followed, or null if there was nothing there
   * distinct enough to recognise again.
   */
  follow(point: { x: number; y: number }, frame: Frame, now: number): Track | null {
    const holding = this.confirmed()
      .filter(
        (t) =>
          point.x >= t.box.x1 && point.x <= t.box.x2 && point.y >= t.box.y1 && point.y <= t.box.y2,
      )
      .sort((a, b) => area(a.box) - area(b.box));

    const chosen = holding[0];
    if (chosen) {
      this.pinned = chosen.id;
      this.remember(chosen, frame);
      this.attending = [chosen.id, ...this.attending.filter((id) => id !== chosen.id)];
      return chosen;
    }

    const half = (this.frameSpan * POINTED_AT) / 2;
    const box = {
      x1: point.x - half,
      y1: point.y - half,
      x2: point.x + half,
      y2: point.y + half,
    };
    const template = print(frame, box);
    if (!template) return null; // nothing here distinct enough to find again

    const track: Track = {
      id: this.nextId++,
      label: "this",
      box,
      velocity: { x: 0, y: 0 },
      score: 1,
      hits: MIN_HITS,
      misses: 0,
      firstSeen: now,
      lastSeen: now,
      lastDetected: now,
      area: area(box) / this.frameArea,
      areaTrend: "steady",
      travelled: 0,
      pace: 0,
      salience: 1,
      background: false,
      template,
      lock: 1,
      lockable: enough(frame, box),
    };
    this.tracks.push(track);
    this.pinned = track.id;
    this.attending = [track.id, ...this.attending];
    return track;
  }

  /** Stop following by hand, and let salience decide again. */
  release() {
    this.pinned = null;
  }

  /** The track someone asked to follow, if it is still here. */
  following(): Track | null {
    return this.tracks.find((t) => t.id === this.pinned) ?? null;
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
    this.at = now;
    return this.confirmed().map((t) => ({ ...t, box: predictBox(t, now) }));
  }

  /** The moment the caller is asking about, so `confirmed` can judge a grace. */
  private at = 0;
  /** The interval between detection passes, as observed rather than assumed. */
  private cadence = 0;
  private lastUpdate = -1;

  /** How long to keep something nothing has confirmed lately. */
  private patience(): number {
    return Math.min(Math.max(MAX_UNSEEN_MS, this.cadence * UNSEEN_PASSES), MAX_PATIENCE_MS);
  }

  /** How long a miss alone is not taken for absence. */
  private grace(): number {
    return Math.min(Math.max(GRACE_MS, this.cadence * GRACE_PASSES), MAX_PATIENCE_MS);
  }

  private confirmed(): Track[] {
    return this.tracks
      /*
       * A detector miss alone is not grounds to stop drawing a box.
       *
       * Three things can keep one: the detector has just seen it, the print still
       * matches, or not long enough has passed to give up on it. The last exists
       * because neither of the first two is a reliable witness to absence — a
       * detector measured to mention only a third of what is in front of it, and
       * a print that says nothing at all about a subject seventeen pixels tall or
       * one that has stepped behind a lamp post. So the lock only ever extends
       * confidence and never cuts it short, and what has really gone is forgotten
       * by retirement rather than by a failed match.
       */
      .filter(
        (t) =>
          t.hits >= MIN_HITS &&
          (t.misses === 0 || t.lock >= LOCK_KEEP || this.at - t.lastDetected < this.grace()),
      )
      .sort((a, b) => b.area - a.area);
  }

  /** Every live track, including ones currently unobserved. */
  all(): Track[] {
    return this.tracks.filter((t) => t.hits >= MIN_HITS);
  }

  reset() {
    this.tracks = [];
    this.nextId = 1;
    this.attending = [];
    this.attendedAt = -1;
    this.pinned = null;
    this.cadence = 0;
    this.lastUpdate = -1;
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
