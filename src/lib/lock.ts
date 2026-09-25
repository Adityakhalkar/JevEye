"use client";

/**
 * Holding on to a thing between detections, by looking at it.
 *
 * A detector costs the better part of a second, so between its answers a box
 * has to be carried forward somehow. Extrapolating along a velocity is a guess,
 * and guesses drift: the box slides off the subject and sits on empty grass,
 * still labelled, still confident. That is the failure this fixes.
 *
 * Each track keeps a small greyscale print of what it was looking at when it
 * was last seen, and every frame the neighbourhood is searched for where that
 * pattern went.
 *
 * Nothing here touches a canvas. The frame is read back once, into plain
 * numbers, and every track is matched against that — because a readback is a
 * stall waiting on the GPU, and doing one per track cost 50ms a frame and
 * slowed the detector to a crawl. One readback and arithmetic is fifteen times
 * cheaper, and pure functions over an array are far easier to be sure of.
 */

/** The print's side, in print pixels. Matching cost is independent of box size. */
const PRINT = 24;
/** How far to look around the expected position, as a share of the box. */
const SEARCH = 0.6;
/** The resampled search window's side, at the same scale as the print. */
const WINDOW = Math.round(PRINT * (1 + 2 * SEARCH));
/** Offsets the print can take inside the window, per axis. */
const OFFSETS = WINDOW - PRINT + 1;
/** Sidelobes are measured outside this radius of the peak, in print pixels. */
const PEAK_RADIUS = 3;

/**
 * A frame's brightness, one number per pixel.
 *
 * `scale` is how many of these pixels there are per unit of box coordinate, so
 * the frame can be read at a finer resolution than the boxes are measured in.
 * That matters more than anything else here: measured on a crowded street, the
 * median watched subject was 37 by 17 pixels, a print of which is mush — and the
 * match failed half the time. On subjects of fifty pixels and up it succeeded
 * every time, at 0.94. Detail is the whole game, and the detector's input size
 * is no reason to throw it away.
 */
export type Frame = { pixels: Float32Array; width: number; height: number; scale: number };
export type Print = { pixels: Float32Array; span: number };
export type Rect = { x1: number; y1: number; x2: number; y2: number };
export type Lock = { rect: Rect; score: number; sharpness: number };

// Fixed sizes, so the working buffers are allocated once rather than per frame.
const window_ = new Float32Array(WINDOW * WINDOW);
const sums = new Float64Array((WINDOW + 1) * (WINDOW + 1));
const squares = new Float64Array((WINDOW + 1) * (WINDOW + 1));
const surface = new Float32Array(OFFSETS * OFFSETS);
const scratch = new Float32Array(WINDOW * WINDOW);

/**
 * Read a canvas once into brightness values.
 *
 * This is the only expensive step, and it happens a single time per frame
 * however many things are being followed.
 *
 * Pass the previous frame back as `reuse` to write into its buffer instead of
 * allocating another: at ten frames a second a fresh buffer per frame is a third
 * of a megabyte of garbage a second, and the collector's pauses land in the
 * middle of the video. Only do that where the old frame is genuinely finished
 * with — a print still being cut from an earlier frame needs its own.
 */
export function frameOf(
  source: HTMLCanvasElement,
  reuse?: Frame | null,
  scale = 1,
): Frame | null {
  const context = source.getContext("2d", { willReadFrequently: true });
  if (!context || !source.width || !source.height) return null;
  const { width, height } = source;
  const { data } = context.getImageData(0, 0, width, height);
  const pixels =
    reuse && reuse.width === width && reuse.height === height
      ? reuse.pixels
      : new Float32Array(width * height);
  for (let i = 0; i < pixels.length; i++) {
    pixels[i] = (data[i * 4] * 77 + data[i * 4 + 1] * 150 + data[i * 4 + 2] * 29) / 256;
  }
  return { pixels, width, height, scale };
}

/** Brightness between pixels, interpolated and clamped at the frame's edge. */
function at(frame: Frame, x: number, y: number): number {
  const { pixels, width, height } = frame;
  const cx = x < 0 ? 0 : x > width - 1 ? width - 1 : x;
  const cy = y < 0 ? 0 : y > height - 1 ? height - 1 : y;
  const x0 = Math.floor(cx);
  const y0 = Math.floor(cy);
  const x1 = x0 + 1 < width ? x0 + 1 : x0;
  const y1 = y0 + 1 < height ? y0 + 1 : y0;
  const fx = cx - x0;
  const fy = cy - y0;
  const top = pixels[y0 * width + x0] * (1 - fx) + pixels[y0 * width + x1] * fx;
  const bottom = pixels[y1 * width + x0] * (1 - fx) + pixels[y1 * width + x1] * fx;
  return top * (1 - fy) + bottom * fy;
}

/** Resample a rectangle of the frame into a square of `side`, into `into`. */
function resample(
  frame: Frame,
  x: number,
  y: number,
  w: number,
  h: number,
  side: number,
  into: Float32Array,
) {
  const { scale } = frame;
  const stepX = (w * scale) / side;
  const stepY = (h * scale) / side;
  for (let j = 0; j < side; j++) {
    const sy = y * scale + (j + 0.5) * stepY;
    for (let i = 0; i < side; i++) {
      into[j * side + i] = at(frame, x * scale + (i + 0.5) * stepX, sy);
    }
  }
}

/**
 * A light [1 2 1] blur, separable, clamped at the edges.
 *
 * Without it the response is a single-pixel spike sitting on noise: findable,
 * but with nothing either side of the peak to interpolate against, so the box
 * can only land on whole print pixels and visibly steps between them. Blurring
 * widens the peak just enough to fit a curve through.
 */
function smooth(pixels: Float32Array, side: number, spare: Float32Array) {
  for (let y = 0; y < side; y++) {
    const row = y * side;
    for (let x = 0; x < side; x++) {
      const l = pixels[row + (x > 0 ? x - 1 : 0)];
      const r = pixels[row + (x < side - 1 ? x + 1 : side - 1)];
      spare[row + x] = (l + 2 * pixels[row + x] + r) / 4;
    }
  }
  for (let x = 0; x < side; x++) {
    for (let y = 0; y < side; y++) {
      const u = spare[(y > 0 ? y - 1 : 0) * side + x];
      const d = spare[(y < side - 1 ? y + 1 : side - 1) * side + x];
      pixels[y * side + x] = (u + 2 * spare[y * side + x] + d) / 4;
    }
  }
}

/**
 * A zero-mean, unit-norm print of a region.
 *
 * Normalising is what makes the match indifferent to the scene getting brighter
 * or darker: only the pattern survives, not the exposure.
 */
export function print(frame: Frame, rect: Rect): Print | null {
  const w = rect.x2 - rect.x1;
  const h = rect.y2 - rect.y1;
  if (w < 4 || h < 4) return null;

  const pixels = new Float32Array(PRINT * PRINT);
  resample(frame, rect.x1, rect.y1, w, h, PRINT, pixels);
  smooth(pixels, PRINT, scratch);

  let sum = 0;
  for (let i = 0; i < pixels.length; i++) sum += pixels[i];
  const mean = sum / pixels.length;
  let energy = 0;
  for (let i = 0; i < pixels.length; i++) {
    pixels[i] -= mean;
    energy += pixels[i] * pixels[i];
  }
  const norm = Math.sqrt(energy);
  if (norm < 1e-3) return null; // a flat patch matches everything and means nothing
  for (let i = 0; i < pixels.length; i++) pixels[i] /= norm;
  return { pixels, span: Math.max(w, h) };
}

/**
 * Whether there are enough real pixels here for a match to mean anything.
 *
 * Measured on a crowded street: on subjects whose shorter side reached fifty
 * pixels the match held at 0.94 and never fell below the threshold, while under
 * thirty it sat at 0.48 — a coin toss. Two source pixels per print pixel is the
 * line, and below it the honest answer is that appearance cannot settle the
 * question, not that the subject has gone.
 */
export function enough(frame: Frame, rect: Rect): boolean {
  const shorter = Math.min(rect.x2 - rect.x1, rect.y2 - rect.y1) * frame.scale;
  return shorter >= PRINT * 2;
}

/** Reused by `verify`, so confirming a print costs no allocation. */
const spot = new Float32Array(PRINT * PRINT);

/**
 * How well a print still matches at exactly this position, without searching.
 *
 * Searching is for things whose box has to land somewhere exact. Most of what is
 * in shot only needs the question answered — is it still there? — and that is a
 * single resample and a dot product, some thirty times cheaper than a search. It
 * is what lets everything in view keep its identity through a detector's
 * dropouts while only the few things on screen pay for a search.
 */
export function verify(frame: Frame, template: Print, rect: Rect): number {
  const w = rect.x2 - rect.x1;
  const h = rect.y2 - rect.y1;
  if (w < 4 || h < 4) return 0;
  resample(frame, rect.x1, rect.y1, w, h, PRINT, spot);
  smooth(spot, PRINT, scratch);
  let sum = 0;
  for (let i = 0; i < spot.length; i++) sum += spot[i];
  const mean = sum / spot.length;
  let energy = 0;
  for (let i = 0; i < spot.length; i++) {
    spot[i] -= mean;
    energy += spot[i] * spot[i];
  }
  const norm = Math.sqrt(energy);
  if (norm < 1e-3) return 0;
  let dot = 0;
  for (let i = 0; i < spot.length; i++) dot += (spot[i] / norm) * template.pixels[i];
  return dot;
}

/** Correlation of two prints: 1 is identical, 0 unrelated, -1 inverted. */
export function correlate(a: Print, b: Print): number {
  let total = 0;
  for (let i = 0; i < a.pixels.length; i++) total += a.pixels[i] * b.pixels[i];
  return total;
}

/**
 * Ease a fresh look into a print, keeping it zero-mean and unit-norm.
 *
 * A print taken at identification and never touched is stale by the time the
 * detector speaks again: measured on real footage, the same subject correlates
 * about 0.63 across a 1.3-second gap, with a long tail below the threshold worth
 * moving a box for — so a subject plainly in view was being declared lost. A
 * print that is never refreshed cannot follow anything that turns.
 *
 * Refreshing it wholesale is the opposite failure, the one this used to guard
 * against: a print recut from where the tracker believes the subject to be
 * follows its own error, a pixel at a time, onto the background. A small
 * fraction eased in, and only when the match was strong enough to be trusted,
 * tracks a subject rotating or changing its lighting while leaving no room for
 * error to compound — a bad look cannot move it, because a bad look is not
 * allowed to contribute.
 */
export function freshen(template: Print, fresh: Print, rate: number): Print {
  const pixels = new Float32Array(template.pixels.length);
  let sum = 0;
  for (let i = 0; i < pixels.length; i++) {
    pixels[i] = template.pixels[i] * (1 - rate) + fresh.pixels[i] * rate;
    sum += pixels[i];
  }
  const mean = sum / pixels.length;
  let energy = 0;
  for (let i = 0; i < pixels.length; i++) {
    pixels[i] -= mean;
    energy += pixels[i] * pixels[i];
  }
  const norm = Math.sqrt(energy);
  if (norm < 1e-6) return template;
  for (let i = 0; i < pixels.length; i++) pixels[i] /= norm;
  return { pixels, span: fresh.span };
}

/** Where a parabola through three samples peaks, in samples either side of the middle. */
function apex(left: number, middle: number, right: number): number {
  const curve = left - 2 * middle + right;
  if (curve >= -1e-9) return 0; // not a peak; don't interpolate off it
  const shift = (0.5 * (left - right)) / curve;
  return shift < -1 ? -1 : shift > 1 ? 1 : shift;
}

/**
 * Find where a print has moved to, searching the whole neighbourhood.
 *
 * The window is resampled once and every offset inside it scored, which has to
 * be exhaustive: the correlation peak is only a pixel or two wide, and a coarse
 * search walks straight past it. Sums and sums-of-squares come from integral
 * images, so each offset's normalisation is a handful of lookups and only the
 * cross-product is actually summed.
 *
 * The window is stretched by the same factor in each direction as the print, so
 * a tall thin subject is matched against pixels squashed exactly as its own
 * print was.
 */
export function relock(frame: Frame, template: Print, expected: Rect): Lock | null {
  const w = expected.x2 - expected.x1;
  const h = expected.y2 - expected.y1;
  if (w < 4 || h < 4) return null;

  const grown = 1 + 2 * SEARCH;
  const windowW = w * grown;
  const windowH = h * grown;
  if (windowW * frame.scale < WINDOW / 2 || windowH * frame.scale < WINDOW / 2) return null;
  const x0 = (expected.x1 + expected.x2) / 2 - windowW / 2;
  const y0 = (expected.y1 + expected.y2) / 2 - windowH / 2;
  resample(frame, x0, y0, windowW, windowH, WINDOW, window_);
  smooth(window_, WINDOW, scratch);

  const stride = WINDOW + 1;
  sums.fill(0);
  squares.fill(0);
  for (let y = 0; y < WINDOW; y++) {
    for (let x = 0; x < WINDOW; x++) {
      const v = window_[y * WINDOW + x];
      const here = (y + 1) * stride + (x + 1);
      sums[here] = v + sums[here - 1] + sums[here - stride] - sums[here - stride - 1];
      squares[here] =
        v * v + squares[here - 1] + squares[here - stride] - squares[here - stride - 1];
    }
  }
  const area = PRINT * PRINT;
  const block = (table: Float64Array, x: number, y: number) =>
    table[(y + PRINT) * stride + x + PRINT] -
    table[y * stride + x + PRINT] -
    table[(y + PRINT) * stride + x] +
    table[y * stride + x];

  let bestScore = -Infinity;
  let bestX = 0;
  let bestY = 0;
  for (let oy = 0; oy < OFFSETS; oy++) {
    for (let ox = 0; ox < OFFSETS; ox++) {
      const total = block(sums, ox, oy);
      const variance = block(squares, ox, oy) - (total * total) / area;
      let score = 0;
      if (variance > 1e-3) {
        // The print is zero-mean, so the window's own mean drops out of the
        // cross-product and only the dot product needs summing.
        let dot = 0;
        for (let ty = 0; ty < PRINT; ty++) {
          const windowRow = (oy + ty) * WINDOW + ox;
          const printRow = ty * PRINT;
          for (let tx = 0; tx < PRINT; tx++) {
            dot += window_[windowRow + tx] * template.pixels[printRow + tx];
          }
        }
        score = dot / Math.sqrt(variance);
      }
      surface[oy * OFFSETS + ox] = score;
      if (score > bestScore) {
        bestScore = score;
        bestX = ox;
        bestY = oy;
      }
    }
  }
  if (!(bestScore > 0)) return null;

  // How far the peak stands above everywhere else. A real match is a spike on a
  // flat field; something that merely resembles the print lifts the whole
  // surface and stands out from it barely at all.
  let sum = 0;
  let squareSum = 0;
  let count = 0;
  for (let oy = 0; oy < OFFSETS; oy++) {
    for (let ox = 0; ox < OFFSETS; ox++) {
      if (Math.abs(ox - bestX) <= PEAK_RADIUS && Math.abs(oy - bestY) <= PEAK_RADIUS) continue;
      const v = surface[oy * OFFSETS + ox];
      sum += v;
      squareSum += v * v;
      count++;
    }
  }
  const mean = count ? sum / count : 0;
  const spread = count ? Math.sqrt(Math.max(squareSum / count - mean * mean, 1e-12)) : 1;

  const scoreAt = (ox: number, oy: number) =>
    ox < 0 || oy < 0 || ox >= OFFSETS || oy >= OFFSETS ? -1 : surface[oy * OFFSETS + ox];
  const shiftX = apex(scoreAt(bestX - 1, bestY), bestScore, scoreAt(bestX + 1, bestY));
  const shiftY = apex(scoreAt(bestX, bestY - 1), bestScore, scoreAt(bestX, bestY + 1));

  const x1 = x0 + (bestX + shiftX) * (windowW / WINDOW);
  const y1 = y0 + (bestY + shiftY) * (windowH / WINDOW);
  return {
    rect: { x1, y1, x2: x1 + w, y2: y1 + h },
    score: bestScore,
    sharpness: (bestScore - mean) / spread,
  };
}
