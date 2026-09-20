import type { ScaleId } from "./scales";
import type { VocabularyId } from "./vocab";

/** What Jev decided the question means, before any pixel is touched. */
export type Plan = {
  /** Probability the question is about something visible in a photograph. */
  onTopic: number;
  /** Set to false when `onTopic` is below the gate; no probes run. */
  proceed: boolean;
  /** Which shape of answer the question wants — and which probes will run. */
  reading: Reading;
  readingConfidence: number;
  vocabulary: VocabularyId;
  vocabularyConfidence: number;
  /** Open-vocabulary detection prompt for the vocabulary's instances. */
  instanceNoun: string;
  /** For `presence`: the one label the question asks about, chosen from the vocabulary. */
  subject: string | null;
  subjectConfidence: number | null;
  /** For `rating`: the scale the question is asking along. */
  scale: ScaleId | null;
  scaleConfidence: number | null;
  usdCost: number;
  inputTokens: number;
};

export type Reading = "only" | "dominant" | "every" | "presence" | "count" | "rating";

export const READINGS: Record<Reading, string> = {
  only: "Which single kind the picture is of, assuming there is only one",
  dominant: "Which kind is the most common one in the picture",
  every: "Which kinds appear in the picture at all",
  presence: "Whether one particular kind is present or absent",
  count: "How many of something there are, or how much of the picture it fills",
  rating:
    "How much of some quality the picture shows, along a scale — how healthy, how blurry, how crowded, how damaged, how well lit",
};

/** Readings that need the label set, and so run the classifier over tiles. */
export const IDENTIFYING: ReadonlySet<Reading> = new Set<Reading>(["only", "dominant", "every"]);

/**
 * A per-label tally over the tiles that hold the subject.
 *
 * `weight` sums the confidences, so a label found faintly in four tiles does not
 * outrank one found clearly in three.
 */
export type Tally = { label: string; count: number; meanConfidence: number; weight: number };

/** Shared by every fact sheet, whatever the reading. */
type Observed = {
  imageSize: { width: number; height: number };
  elapsedMs: number;
  /** Free-text presence probes, or null where the detector abstained. */
  context: Record<string, number | null>;
};

/** How much of the picture holds the subject, as a distribution over tiles. */
type Coverage = {
  grid: { cols: number; rows: number };
  tilesExamined: number;
  tilesWithSubject: number;
  tilesLow: number;
  tilesHigh: number;
};

export type IdentifyFacts = Observed &
  Coverage & {
    kind: "identify";
    tallies: Tally[];
    /** Tiles whose best label fell below the choose bar. */
    unknownTiles: number;
    /** Whole-image reading, used when no tile held the subject. */
    wholeImage: { label: string; confidence: number } | null;
    /** Share of identified weight held by the leading label, 0..1. */
    dominantShare: number | null;
    /** Whether a fitted probe named the tiles, or the zero-shot text classifier. */
    classifier: "probe" | "zero-shot";
    /** The probe's measured held-out accuracy and ECE, when a probe answered. */
    classifierAccuracy: number | null;
    classifierEce: number | null;
  };

export type PresenceFacts = Observed &
  Coverage & {
    kind: "presence";
    subject: string;
    /**
     * Probability of the subject when every label in the vocabulary competes.
     *
     * Not a yes/no score against filler sentences: CLIP puts any concrete
     * sentence far above abstract ones, so "is there an airplane" beat
     * "something else entirely" on a photograph of a dog. Real alternatives are
     * what make an absence measurable.
     */
    subjectProbability: number;
    /** Where the subject placed among the vocabulary, 1 = most likely. */
    subjectRank: number;
    /** What the classifier actually thinks is in the picture. */
    topLabels: Array<{ label: string; p: number }>;
    /** Tiles whose best label was the subject, out of those actually checked. */
    tilesMatchingSubject: number;
    tilesChecked: number;
    classifier: "probe" | "zero-shot";
    /** How sure Jev was that this subject is what the question asked about. */
    subjectConfidence: number | null;
  };

export type CountFacts = Observed & Coverage & { kind: "count"; noun: string };

export type RatingFacts = Observed & {
  kind: "rating";
  scale: ScaleId;
  levels: readonly string[];
  /** 0 = lowest level, 1 = highest. */
  position: number;
  /** Null when the scale could not be read reliably. */
  confidence: number | null;
};

export type FactSheet = IdentifyFacts | PresenceFacts | CountFacts | RatingFacts;

export type Judgment = {
  /** The label, the verdict, or the level name — whatever the reading asked for. */
  answer: string;
  confidence: number;
  /** Present for readings judged as a choice. */
  probabilities?: Record<string, number>;
  /** Present for `identify`: whether more than one kind is meaningfully there. */
  mixed?: number;
  /** Present for `presence`: Jev's own probability that the subject is there. */
  present?: number;
  support: number;
  supportLegend: Record<string, unknown>;
  usdCost: number;
  inputTokens: number;
};
