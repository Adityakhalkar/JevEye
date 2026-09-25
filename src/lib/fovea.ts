"use client";

/**
 * Looking closely at whatever the tracker is least sure about.
 *
 * The detector is the one expensive thing here — some seven hundred milliseconds
 * a pass — and it was being spent the same way every time: the whole frame,
 * shrunk to fit, whatever was happening in it. That is a strange way to spend a
 * scarce resource, because the system already knows where its picture of the
 * world is failing. The lock reports a confidence for every subject ten times a
 * second for about a millisecond, so doubt is the cheapest signal in the system
 * and it was being thrown away.
 *
 * The measurement that forced this: on real footage the median subject worth
 * watching is 37 by 17 pixels, and appearance matching is a coin toss under
 * thirty pixels while it is near-perfect over fifty. The frame is 336 across and
 * the video is 640 — the detail exists, it is being discarded to make the
 * detector affordable.
 *
 * So passes alternate. A wide one keeps the whole scene in view. A close one
 * crops the most doubtful region straight from the video at its own resolution
 * and hands the detector that instead, which costs exactly the same and spends
 * it on four times the pixels where they are needed. Doubt fades with time, so
 * the close look wanders rather than fixating, and a region just examined stops
 * asking to be examined again.
 *
 * This is a budget, not a heuristic: attention is finite, and what it buys is
 * decided by where the system is wrong rather than by where it happens to be
 * pointing.
 */

import type { Box } from "./track.ts";

/** The doubt map's resolution. Coarse on purpose: this picks a region, not a pixel. */
const COLUMNS = 6;
const ROWS = 4;

/**
 * How much of the frame a close look covers, per side.
 *
 * Measured, not chosen: tightening it to a quarter made things worse — the match
 * fell from 0.55 to 0.41 and failures rose from 49% to 75% — because a narrow
 * crop covers fewer subjects per pass and loses the context around them, while
 * the subjects themselves barely grew (33px against 35px). Two fifths is where
 * detail and coverage balance.
 */
export const FOVEA_SHARE = 0.4;

/**
 * How much a second of not having looked somewhere is worth, against doubt.
 *
 * Curiosity, as a number. Without it the close look would fixate on whatever is
 * currently hardest and never check anywhere else, so something walking in
 * unnoticed would stay unnoticed. With it, a region nobody has examined for long
 * enough eventually outbids a familiar problem — and because the term is time
 * since a *close look*, a region that keeps disappointing does not keep winning.
 */
const CURIOSITY = 0.6;

/** How long doubt takes to halve, so what counts is being unsure lately. */
const FORGET_MS = 2000;

/** Doubt carried by something the tracker cannot recognise at all. */
const UNRECOGNISABLE = 1;

/** What can be said about one subject, as far as doubt is concerned. */
export type Held = { box: Box; lock: number; lockable: boolean };

export class Doubt {
  private cells = new Float32Array(COLUMNS * ROWS);
  /** When each cell was last examined closely. */
  private seen = new Float64Array(COLUMNS * ROWS);
  private at = 0;

  private cell(box: Box, frame: { width: number; height: number }): number {
    const cx = ((box.x1 + box.x2) / 2 / frame.width) * COLUMNS;
    const cy = ((box.y1 + box.y2) / 2 / frame.height) * ROWS;
    const column = Math.min(COLUMNS - 1, Math.max(0, Math.floor(cx)));
    const row = Math.min(ROWS - 1, Math.max(0, Math.floor(cy)));
    return row * COLUMNS + column;
  }

  /**
   * Take in what the tracker currently does and does not understand.
   *
   * Doubt is forgotten as it ages, so this reflects being unsure *lately* rather
   * than a tally of every difficulty ever had. Weighted towards what cannot be
   * checked at all: a subject too few pixels across to recognise is exactly what
   * a closer look would fix, while a large one held confidently gains nothing.
   */
  observe(subjects: Held[], frame: { width: number; height: number }, now: number) {
    if (this.at) {
      const fade = Math.pow(0.5, Math.min(now - this.at, 10_000) / FORGET_MS);
      for (let i = 0; i < this.cells.length; i++) this.cells[i] *= fade;
    }
    this.at = now;
    for (const subject of subjects) {
      const amount = Math.max(0, 1 - subject.lock) + (subject.lockable ? 0 : UNRECOGNISABLE);
      if (amount > 0) this.cells[this.cell(subject.box, frame)] += amount;
    }
  }

  /** How strongly each region is asking to be looked at. */
  private appetite(index: number, now: number): number {
    const since = this.seen[index] ? (now - this.seen[index]) / 1000 : 30;
    return this.cells[index] + CURIOSITY * since;
  }

  /** Where the close look should go, as a region of the frame. */
  where(frame: { width: number; height: number }, now: number, share = FOVEA_SHARE): Box {
    let best = -Infinity;
    let pick = 0;
    for (let i = 0; i < this.cells.length; i++) {
      const appetite = this.appetite(i, now);
      if (appetite > best) {
        best = appetite;
        pick = i;
      }
    }
    const column = pick % COLUMNS;
    const row = Math.floor(pick / COLUMNS);
    const w = frame.width * share;
    const h = frame.height * share;
    const cx = ((column + 0.5) / COLUMNS) * frame.width;
    const cy = ((row + 0.5) / ROWS) * frame.height;
    const x1 = Math.min(Math.max(cx - w / 2, 0), Math.max(0, frame.width - w));
    const y1 = Math.min(Math.max(cy - h / 2, 0), Math.max(0, frame.height - h));
    return { x1, y1, x2: x1 + w, y2: y1 + h };
  }

  /** Somewhere has just been examined, so it stops asking for a while. */
  looked(region: Box, frame: { width: number; height: number }, now: number) {
    for (let row = 0; row < ROWS; row++) {
      for (let column = 0; column < COLUMNS; column++) {
        const x = ((column + 0.5) / COLUMNS) * frame.width;
        const y = ((row + 0.5) / ROWS) * frame.height;
        if (x < region.x1 || x > region.x2 || y < region.y1 || y > region.y2) continue;
        const index = row * COLUMNS + column;
        this.cells[index] = 0;
        this.seen[index] = now;
      }
    }
  }

  /** How unsure the system is overall, for reporting rather than deciding. */
  unsure(): number {
    let sum = 0;
    for (const cell of this.cells) sum += cell;
    return sum;
  }

  reset() {
    this.cells.fill(0);
    this.seen.fill(0);
    this.at = 0;
  }
}

/** Put a box found inside a close look back into the frame's own coordinates. */
export function outward(box: Box, region: Box, looked: { width: number; height: number }): Box {
  const scaleX = (region.x2 - region.x1) / looked.width;
  const scaleY = (region.y2 - region.y1) / looked.height;
  return {
    x1: region.x1 + box.x1 * scaleX,
    y1: region.y1 + box.y1 * scaleY,
    x2: region.x1 + box.x2 * scaleX,
    y2: region.y1 + box.y2 * scaleY,
  };
}
