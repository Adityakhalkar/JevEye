/**
 * Frames made of nothing but numbers, for tests.
 *
 * Matching and tracking are worth testing against scenes whose right answer is
 * known to the pixel, which no recorded video can give. Since the matching reads
 * plain brightness values rather than a canvas, a scene is just a function.
 */
import type { Frame } from "./lock.ts";

/** The frame size every mock scene uses. */
export const FRAME = { width: 320, height: 240 };
/** The mock subject's side, in pixels. */
export const SIZE = 40;

export function frameOfPaint(
  paint: (x: number, y: number) => number,
  width = FRAME.width,
  height = FRAME.height,
): Frame {
  const pixels = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) pixels[y * width + x] = paint(x, y);
  }
  return { pixels, width, height };
}

/**
 * A subject with structure, on a plain field.
 *
 * A featureless blob is the one thing template matching cannot follow, and
 * correctly refuses to, so this subject has internal detail the way a real one
 * does: a bright body with two darker markings placed off-centre, which makes
 * exactly one alignment correct.
 */
export function sceneWith(x: number, y: number, exposure = 0, flatten = 0): Frame {
  return frameOfPaint((px, py) => {
    const lx = px - x;
    const ly = py - y;
    if (!(lx >= 0 && lx < SIZE && ly >= 0 && ly < SIZE)) return 90 + exposure;
    if (lx >= 4 && lx < 14 && ly >= 4 && ly < 14) return 40 + exposure + flatten;
    if (lx >= 24 && lx < 32 && ly >= 26 && ly < 34) return 150 + exposure;
    return 230 + exposure - flatten;
  });
}

export const boxAt = (x: number, y: number) => ({ x1: x, y1: y, x2: x + SIZE, y2: y + SIZE });

/** An empty field: the subject has left. */
export const emptyScene = () => frameOfPaint(() => 90);
