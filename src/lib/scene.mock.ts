/**
 * Canvases made of nothing but pixels, for tests.
 *
 * Matching and tracking are worth testing against scenes whose right answer is
 * known to the pixel, which a real canvas cannot provide. This stands in for the
 * one drawing operation the code uses — a scaled sub-rectangle blit — with
 * nearest-neighbour sampling, which is harsher than a browser's filtered
 * downscale and so exercises the harder case.
 */

export type Stub = {
  width: number;
  height: number;
  buffer: Uint8ClampedArray;
  blit: { source: Stub; sx: number; sy: number; sw: number; sh: number } | null;
  getContext: () => CanvasRenderingContext2D;
};

export function canvasOf(
  width: number,
  height: number,
  paint: (x: number, y: number) => number,
): Stub {
  const buffer = new Uint8ClampedArray(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) buffer[y * width + x] = paint(x, y);
  }
  const stub: Stub = {
    width,
    height,
    buffer,
    blit: null,
    getContext: () =>
      ({
        drawImage: (source: Stub, sx: number, sy: number, sw: number, sh: number) => {
          stub.blit = { source, sx, sy, sw, sh };
        },
        getImageData: (_x: number, _y: number, w: number, h: number) => {
          const { source, sx, sy, sw, sh } = stub.blit!;
          const data = new Uint8ClampedArray(w * h * 4);
          for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
              const px = Math.min(source.width - 1, Math.max(0, Math.round(sx + (x / w) * sw)));
              const py = Math.min(source.height - 1, Math.max(0, Math.round(sy + (y / h) * sh)));
              const v = source.buffer[py * source.width + px];
              const i = (y * w + x) * 4;
              data[i] = data[i + 1] = data[i + 2] = v;
              data[i + 3] = 255;
            }
          }
          return { data };
        },
      }) as unknown as CanvasRenderingContext2D,
  };
  return stub;
}

export const asCanvas = (stub: Stub) => stub as unknown as HTMLCanvasElement;

/** The frame size every mock scene uses. */
export const FRAME = { width: 320, height: 240 };
/** The mock subject's side, in pixels. */
export const SIZE = 40;

/**
 * A subject with structure, on a plain field.
 *
 * A featureless blob is the one thing template matching cannot follow, and
 * correctly refuses to, so this subject has internal detail the way a real one
 * does: a bright body with two darker markings placed off-centre, which makes
 * exactly one alignment correct.
 */
export function sceneWith(x: number, y: number, exposure = 0, flatten = 0): Stub {
  return canvasOf(FRAME.width, FRAME.height, (px, py) => {
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
export const emptyScene = () => canvasOf(FRAME.width, FRAME.height, () => 90);
