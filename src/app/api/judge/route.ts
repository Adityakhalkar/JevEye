/**
 * Jev judges the fact sheet. It never sees the image — only what the vision
 * layer reported, with every confidence still attached, so a shaky observation
 * stays shaky all the way to the answer.
 */
import { choice, noul, score } from "@typesafe-ai/sdk";
import { NextResponse } from "next/server";

import { jev, jevError, usdCost } from "@/lib/jev";
import { READINGS, type FactSheet, type Judgment, type Reading } from "@/lib/types";

/** Jev's way of declining to name a kind when the observations do not support one. */
export const UNDECIDED = "cannot tell from these observations";

/** The fact sheet as Jev reads it: numbers with their uncertainty intact. */
function observations(facts: FactSheet) {
  return {
    how_the_picture_was_examined: `cut into a ${facts.grid.cols}×${facts.grid.rows} grid of ${facts.tilesExamined} tiles, each looked at separately`,
    tiles_holding_the_subject: `most likely ${facts.tilesWithSubject} of ${facts.tilesExamined}, and between ${facts.tilesLow} and ${facts.tilesHigh} with 90% probability`,
    identified_by_kind: facts.tallies.map(
      (t) =>
        `${t.label}: in ${t.count} tiles, mean confidence ${t.meanConfidence.toFixed(2)}, combined weight ${t.weight.toFixed(2)}`,
    ),
    strongest_kind_by_weight: facts.tallies[0]?.label ?? "none",
    tiles_the_classifier_could_not_name: facts.unknownTiles,
    whole_image_reading: facts.wholeImage
      ? `${facts.wholeImage.label} at confidence ${facts.wholeImage.confidence.toFixed(2)}`
      : "not attempted, or too unreliable to report",
    share_of_total_weight_held_by_the_leading_kind:
      facts.dominantShare === null ? "nothing identified" : facts.dominantShare.toFixed(2),
    other_observations: Object.fromEntries(
      Object.entries(facts.context).map(([k, v]) => [
        k,
        v === null ? "unknown — the detector abstained" : v.toFixed(2),
      ]),
    ),
  };
}

export async function POST(request: Request) {
  let question: string;
  let facts: FactSheet;
  let reading: Reading;
  try {
    ({ question, facts, reading } = await request.json());
  } catch {
    return NextResponse.json({ error: "Body must be JSON." }, { status: 400 });
  }

  const candidates = facts?.tallies?.map((t) => t.label) ?? [];
  if (facts?.wholeImage && !candidates.includes(facts.wholeImage.label)) {
    candidates.push(facts.wholeImage.label);
  }
  if (candidates.length === 0) {
    return NextResponse.json(
      { error: "The vision layer identified nothing, so there is nothing to judge." },
      { status: 422 },
    );
  }

  /**
   * An explicit way out. Without it a single surviving candidate makes the
   * choice trivial and Jev reports confidence 1.00 for a label backed by one
   * shaky observation.
   */
  const criteria: Record<string, string | null> = Object.fromEntries(
    candidates.map((label) => [label, null]),
  );
  criteria[UNDECIDED] =
    "The observations are too thin, too few or too uncertain to name one kind.";

  try {
    const { answers, usage } = await jev().systemOne({
      state: {
        question,
        what_the_question_asks_for: READINGS[reading],
        observations: observations(facts),
        caveat:
          "These confidences come from an uncalibrated image model and are probably too high. Tile counts measure how much of the picture a kind covers, not how many individual ones there are.",
      },
      questions: {
        answer: choice(
          "Given only these observations, which label best answers the question?",
          criteria,
        ),
        mixed: noul("Do the observations show more than one kind covering a meaningful part of the picture?"),
        support: score("How well do these observations support a single confident answer?", [
          "Barely at all — too few tiles, or confidences too low to lean on",
          "Somewhat — a leading kind, but with real ambiguity",
          "Strongly — many consistent tiles behind one kind",
        ]),
      },
    });

    const judgment: Judgment = {
      answer: answers.answer.choice,
      confidence: answers.answer.confidence,
      probabilities: answers.answer.probabilities as Record<string, number>,
      mixed: answers.mixed.noul,
      support: answers.support.score,
      supportLegend: answers.support.legend as Record<string, unknown>,
      inputTokens: usage.input_tokens,
      usdCost: usdCost(usage.input_tokens),
    };
    return NextResponse.json(judgment);
  } catch (error) {
    const { message, status } = jevError(error);
    return NextResponse.json({ error: message }, { status });
  }
}
