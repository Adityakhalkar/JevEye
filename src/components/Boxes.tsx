"use client";

import type { Found } from "@/lib/types";

/**
 * The detector's boxes, drawn over the picture they came from.
 *
 * Coordinates arrive in the prepared image's pixels, so the SVG carries that
 * size as its viewBox and scales with whatever the image is displayed at. Only
 * the detector produces these: the classifier paths have no boxes to draw, and
 * inventing some would be the interface claiming more than the model knows.
 */
/** Beyond this the picture is more label than picture. */
const MOST_BOXES = 6;

export function Boxes({
  found,
  size,
  limit = MOST_BOXES,
  highlight,
  looking,
}: {
  found: Found[];
  size: { width: number; height: number };
  limit?: number;
  /** The id, appearing in a label as "#n", that someone asked to follow. */
  highlight?: number;
  /**
   * The region the detector last examined closely.
   *
   * Worth showing rather than hiding: it is the system deciding where it is least
   * sure, and watching it move is watching it think.
   */
  looking?: { x1: number; y1: number; x2: number; y2: number } | null;
}) {
  if (found.length === 0 && !looking) return null;

  /**
   * The strongest few, not everything.
   *
   * A wide scene can hold a dozen true detections — tents, parked cars, a
   * handbag — and drawing them all buries the subject under its own labels.
   */
  const shown = found
    .slice()
    .sort((a, b) => b.score - a.score)
    .slice(0, limit);

  // Text is sized in viewBox units, which scale with however wide the picture
  // is displayed, so this stays around 13px on screen rather than ballooning.
  const fontSize = size.width / 34;
  const pad = fontSize * 0.3;

  return (
    <svg
      viewBox={`0 0 ${size.width} ${size.height}`}
      className="pointer-events-none absolute inset-0 h-full w-full"
      aria-hidden
    >
      {looking && (
        <g>
          <rect
            x={looking.x1}
            y={looking.y1}
            width={looking.x2 - looking.x1}
            height={looking.y2 - looking.y1}
            fill="none"
            stroke="var(--color-ink-3)"
            strokeWidth={fontSize * 0.08}
            strokeDasharray={`${fontSize * 0.5} ${fontSize * 0.4}`}
            rx={fontSize * 0.2}
          />
          <text
            x={looking.x1 + pad}
            y={looking.y2 - pad * 2}
            fontSize={fontSize * 0.8}
            fontFamily="var(--font-mono)"
            fill="var(--color-ink-3)"
          >
            looking closely
          </text>
        </g>
      )}
      {shown.map((f, i) => {
        // The one thing someone chose is drawn to stand out from the ones the
        // system chose for itself, because which is which is worth seeing.
        const chosen = highlight !== undefined && f.label.includes(`#${highlight}`);
        const label = chosen
          ? `following ${f.label}`
          : `${f.label} ${Math.round(f.score * 100)}%`;
        const labelW = label.length * fontSize * 0.6 + pad * 2;
        const above = f.box.y1 > fontSize * 1.8;
        const labelY = above ? f.box.y1 - fontSize * 1.6 : f.box.y1;
        return (
          <g key={`${f.label}-${i}`}>
            <rect
              x={f.box.x1}
              y={f.box.y1}
              width={f.box.x2 - f.box.x1}
              height={f.box.y2 - f.box.y1}
              fill="none"
              stroke={chosen ? "#f4c542" : "var(--color-trace)"}
              strokeWidth={fontSize * (chosen ? 0.24 : 0.14)}
              rx={fontSize * 0.2}
            />
            <rect
              x={f.box.x1}
              y={labelY}
              width={labelW}
              height={fontSize * 1.6}
              fill={chosen ? "#f4c542" : "var(--color-trace)"}
              rx={fontSize * 0.2}
            />
            <text
              x={f.box.x1 + pad}
              y={labelY + fontSize * 1.15}
              fontSize={fontSize}
              fontFamily="var(--font-mono)"
              fill="#0e1013"
            >
              {label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

/** A plain switch, because this is the one control a visitor is likely to touch. */
export function Toggle({
  checked,
  onChange,
  children,
}: {
  checked: boolean;
  onChange: (next: boolean) => void;
  children: React.ReactNode;
}) {
  return (
    <label className="flex cursor-pointer select-none items-center gap-2 text-neutral-400">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="h-4 w-4 accent-[var(--color-trace)]"
      />
      {children}
    </label>
  );
}
