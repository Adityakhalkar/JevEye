import type { VocabularyId } from "./vocab";

/** What Jev decided the question means, before any pixel is touched. */
export type Plan = {
  /** Probability the question is about something visible in a photograph. */
  onTopic: number;
  /** Set to false when `onTopic` is below the gate; no probes run. */
  proceed: boolean;
  vocabulary: VocabularyId;
  vocabularyConfidence: number;
  /** Which shape of answer the question wants. */
  reading: Reading;
  readingConfidence: number;
  /** Open-vocabulary detection prompt, chosen from the vocabulary's labels. */
  instanceNoun: string;
  usdCost: number;
  inputTokens: number;
};

export type Reading = "dominant" | "only" | "every" | "presence" | "count";

export const READINGS: Record<Reading, string> = {
  dominant: "Which kind is the most common one in the picture",
  only: "Which single kind the picture is of, assuming there is only one",
  every: "Which kinds appear in the picture at all",
  presence: "Whether one particular kind is present or absent",
  count: "How many of something there are",
};

/**
 * A per-label tally over the tiles that hold the subject.
 *
 * `weight` sums the confidences, so a label found faintly in four tiles does not
 * outrank one found clearly in three.
 */
export type Tally = { label: string; count: number; meanConfidence: number; weight: number };

/** Everything the vision layer saw, in the form Jev reads it. */
export type FactSheet = {
  grid: { cols: number; rows: number };
  tilesExamined: number;
  /** Most likely number of tiles holding the subject. */
  tilesWithSubject: number;
  tilesLow: number;
  tilesHigh: number;
  tallies: Tally[];
  /** Tiles whose best label fell below the choose floor. */
  unknownTiles: number;
  /** Whole-image reading, used when no tile held the subject. */
  wholeImage: { label: string; confidence: number } | null;
  /** Calibrated presence probabilities for context, or null when abstained. */
  context: Record<string, number | null>;
  /** Share of identified tiles held by the leading label, 0..1. */
  dominantShare: number | null;
  imageSize: { width: number; height: number };
  elapsedMs: number;
};

export type Judgment = {
  answer: string;
  confidence: number;
  probabilities: Record<string, number>;
  mixed: number;
  support: number;
  supportLegend: Record<string, unknown>;
  usdCost: number;
  inputTokens: number;
};
