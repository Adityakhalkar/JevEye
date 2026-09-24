"use client";

/**
 * How much the picture changed, without asking a model.
 *
 * The live gate only needs to know whether the view moved. That was being
 * answered by a CLIP embedding at 162 ms a frame, on the main thread, which is
 * most of a sample's budget spent to compute a single number — and it is the
 * blocking that makes a video stutter, not the throughput.
 *
 * A downscaled grey thumbnail answers the same question in about a millisecond.
 * It is a cruder signal: a light switch registers as change where a semantic
 * embedding would shrug. For deciding when to look harder, crude and instant
 * beats considered and late.
 */

const WIDE = 32;
const HIGH = 24;

export type Thumb = Uint8Array;

/** A tiny greyscale copy of the frame, for comparing against the next one. */
export function thumbnail(source: HTMLCanvasElement, scratch: HTMLCanvasElement): Thumb {
  scratch.width = WIDE;
  scratch.height = HIGH;
  const context = scratch.getContext("2d", { willReadFrequently: true });
  if (!context) return new Uint8Array(WIDE * HIGH);
  context.drawImage(source, 0, 0, WIDE, HIGH);
  const { data } = context.getImageData(0, 0, WIDE, HIGH);
  const grey = new Uint8Array(WIDE * HIGH);
  for (let i = 0; i < grey.length; i++) {
    // Rec. 601 luma: green carries most of perceived brightness.
    grey[i] = (data[i * 4] * 77 + data[i * 4 + 1] * 150 + data[i * 4 + 2] * 29) >> 8;
  }
  return grey;
}

/**
 * Mean absolute difference between two thumbnails, as a fraction of full scale.
 *
 * Zero is an identical picture. Ordinary handheld wobble sits near 0.01, a
 * subject crossing the frame around 0.05, and a cut well above 0.1.
 */
export function difference(a: Thumb, b: Thumb): number {
  if (a.length !== b.length || a.length === 0) return 1;
  let total = 0;
  for (let i = 0; i < a.length; i++) total += Math.abs(a[i] - b[i]);
  return total / a.length / 255;
}
