"use client";

/**
 * JevEye Live: judging a moving scene instead of a still one.
 *
 * Perception and judgment run at different speeds, and the gap between them is
 * the whole design. Embedding a frame takes about 15ms; a Jev round trip takes
 * about 1.1 seconds. So frames are embedded continuously and Jev is asked only
 * when the scene has actually changed — the same shape as sending arithmetic to
 * a relational engine and only the judgments to a semantic model, moved from
 * rows to time.
 *
 * The gate is free: consecutive frame embeddings are unit vectors, so their
 * cosine says how much the view moved. No motion estimation, no second model.
 */
import type { Choice } from "./vision";

/** One look at the scene. */
export type Sample = {
  at: number;
  embed: Float32Array;
  /** Most likely labels for this frame, strongest first. */
  top: Array<{ label: string; p: number }>;
  /** Whether the classifier was willing to name anything at all. */
  named: boolean;
};

/** A label's behaviour across the window, which is what a single frame cannot show. */
export type Subject = {
  label: string;
  /** Fraction of samples in which this was the leading label. */
  seenFraction: number;
  secondsSinceFirstSeen: number;
  secondsSinceLastSeen: number;
  meanProbability: number;
  /** Comparing the window's halves: what a still image has no way to report. */
  trend: "rising" | "falling" | "steady";
};

export type LiveFacts = {
  windowSeconds: number;
  samples: number;
  samplesPerSecond: number;
  subjects: Subject[];
  /** How far the view has moved, as cosine distance (0 = identical). */
  changeSinceLastJudgment: number;
  changeBetweenSamples: number;
  /** How many different labels led at some point — a proxy for churn. */
  distinctLeaders: number;
  /** Samples the classifier refused to name, out of the window. */
  unnamedSamples: number;
};

const WINDOW_MS = 9000;
/** Cosine distance from the frame Jev last saw that counts as a new situation. */
export const CHANGE_THRESHOLD = 0.12;
/** Jev takes ~1.1s; asking faster than this just queues requests behind itself. */
export const MIN_JUDGMENT_GAP_MS = 2500;
/** Ask anyway this often, so a still scene still gets confirmed. */
export const HEARTBEAT_MS = 20000;

/**
 * A rolling window of samples, and the decision of when Jev should look.
 *
 * Keeps only what the last `WINDOW_MS` covers: a live view has no use for a
 * history it will never judge, and the embeddings are the largest thing here.
 */
export class LiveWindow {
  private samples: Sample[] = [];
  private lastJudgedEmbed: Float32Array | null = null;
  private lastJudgedAt = 0;
  private lastLeader: string | null = null;

  push(sample: Sample) {
    this.samples.push(sample);
    const cutoff = sample.at - WINDOW_MS;
    while (this.samples.length > 0 && this.samples[0].at < cutoff) this.samples.shift();
  }

  get size() {
    return this.samples.length;
  }

  get latest(): Sample | null {
    return this.samples.at(-1) ?? null;
  }

  /** Cosine distance between the newest frame and the one Jev last judged. */
  changeSinceJudgment(similarity: (a: Float32Array, b: Float32Array) => number): number {
    const latest = this.latest;
    if (!latest || !this.lastJudgedEmbed) return 1;
    return 1 - similarity(latest.embed, this.lastJudgedEmbed);
  }

  /**
   * Whether to spend a Jev call now.
   *
   * Three reasons, in order of how much they mean: the view moved, the leading
   * label changed, or nothing has been confirmed in a while. The rate limit
   * comes first, because none of the reasons are worth queueing behind a call
   * that has not come back yet.
   */
  shouldJudge(
    now: number,
    similarity: (a: Float32Array, b: Float32Array) => number,
  ): { judge: boolean; reason: string } {
    const latest = this.latest;
    if (!latest || this.samples.length < 3) return { judge: false, reason: "warming up" };
    // The first look is not rate-limited against a judgment that never happened.
    if (!this.lastJudgedEmbed) return { judge: true, reason: "first look" };
    if (now - this.lastJudgedAt < MIN_JUDGMENT_GAP_MS) return { judge: false, reason: "too soon" };

    const change = this.changeSinceJudgment(similarity);
    if (change >= CHANGE_THRESHOLD) {
      return { judge: true, reason: `the view moved (${change.toFixed(2)})` };
    }
    const leader = latest.top[0]?.label ?? null;
    if (leader && leader !== this.lastLeader) {
      return { judge: true, reason: `now leading: ${leader}` };
    }
    if (now - this.lastJudgedAt >= HEARTBEAT_MS) {
      return { judge: true, reason: "nothing has changed for a while" };
    }
    return { judge: false, reason: "settled" };
  }

  /** Record that Jev has seen the current frame, so change is measured from here. */
  markJudged(now: number) {
    const latest = this.latest;
    if (!latest) return;
    this.lastJudgedEmbed = latest.embed;
    this.lastJudgedAt = now;
    this.lastLeader = latest.top[0]?.label ?? null;
  }

  /**
   * What the window says, in the form Jev reads.
   *
   * Everything here is a fact about the window rather than about a frame:
   * presence over time, a trend across its halves, how much the view moved.
   * A per-frame detector can report none of it.
   */
  facts(
    now: number,
    similarity: (a: Float32Array, b: Float32Array) => number,
  ): LiveFacts {
    const samples = this.samples;
    const span = samples.length > 1 ? (samples.at(-1)!.at - samples[0].at) / 1000 : 0;

    const leaders = new Map<string, { hits: number; sum: number; first: number; last: number }>();
    let unnamed = 0;
    for (const s of samples) {
      const top = s.top[0];
      if (!s.named || !top) {
        unnamed += 1;
        continue;
      }
      const e = leaders.get(top.label) ?? { hits: 0, sum: 0, first: s.at, last: s.at };
      e.hits += 1;
      e.sum += top.p;
      e.first = Math.min(e.first, s.at);
      e.last = Math.max(e.last, s.at);
      leaders.set(top.label, e);
    }

    // The window's halves, so a direction can be reported rather than a value.
    const mid = samples.length > 0 ? samples[0].at + (span * 1000) / 2 : now;
    const meanFor = (label: string, half: "early" | "late") => {
      const part = samples.filter((s) =>
        half === "early" ? s.at < mid : s.at >= mid,
      );
      const hits = part.filter((s) => s.top[0]?.label === label);
      if (hits.length === 0) return 0;
      return hits.reduce((a, s) => a + (s.top[0]?.p ?? 0), 0) / part.length;
    };

    const subjects: Subject[] = [...leaders.entries()]
      .map(([label, e]) => {
        const early = meanFor(label, "early");
        const late = meanFor(label, "late");
        const delta = late - early;
        return {
          label,
          seenFraction: e.hits / Math.max(samples.length, 1),
          secondsSinceFirstSeen: (now - e.first) / 1000,
          secondsSinceLastSeen: (now - e.last) / 1000,
          meanProbability: e.sum / e.hits,
          trend: delta > 0.08 ? "rising" : delta < -0.08 ? "falling" : "steady",
        } satisfies Subject;
      })
      .sort((a, b) => b.seenFraction - a.seenFraction || b.meanProbability - a.meanProbability);

    let stepped = 0;
    for (let i = 1; i < samples.length; i++) {
      stepped += 1 - similarity(samples[i].embed, samples[i - 1].embed);
    }

    return {
      windowSeconds: Number(span.toFixed(1)),
      samples: samples.length,
      samplesPerSecond: span > 0 ? Number((samples.length / span).toFixed(1)) : 0,
      subjects,
      changeSinceLastJudgment: Number(this.changeSinceJudgment(similarity).toFixed(3)),
      changeBetweenSamples:
        samples.length > 1 ? Number((stepped / (samples.length - 1)).toFixed(3)) : 0,
      distinctLeaders: leaders.size,
      unnamedSamples: unnamed,
    };
  }
}

/** Turn a classifier result into a sample, keeping only what the window needs. */
export function toSample(at: number, embed: Float32Array, choice: Choice): Sample {
  return {
    at,
    embed,
    top: choice.probabilities.slice(0, 3),
    named: !choice.unknown,
  };
}
