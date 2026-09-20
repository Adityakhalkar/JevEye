/**
 * Jev judges the fact sheet. It never sees the image — only what the vision
 * layer reported, with every confidence still attached, so a shaky observation
 * stays shaky all the way to the answer.
 *
 * Each reading produces a different shape of evidence and so gets different
 * questions: naming a kind is a choice, checking for one thing is a noul,
 * rating is a score. The one constant is that Jev is always told how thin the
 * evidence was.
 */
import { choice, noul, score } from "@typesafe-ai/sdk";
import { NextResponse } from "next/server";

import { jev, jevError, usdCost } from "@/lib/jev";
import { SCALES } from "@/lib/scales";
import { READINGS, type FactSheet, type Judgment, type Reading } from "@/lib/types";

/** Jev's way of declining to name a kind when the observations do not support one. */
export const UNDECIDED = "cannot tell from these observations";

const CAVEAT =
  "These confidences come from an uncalibrated image model and are probably too high. Tile counts measure how much of the picture something covers, not how many individual ones there are.";

/** The three support levels, shared by every reading. */
const SUPPORT = [
  "Barely at all — too little was observed, or the confidences are too low to lean on",
  "Somewhat — a leading reading, but with real ambiguity",
  "Strongly — consistent observations pointing one way",
] as const;

const coverageLine = (f: Extract<FactSheet, { grid: unknown }>) =>
  `the picture was cut into a ${f.grid.cols}×${f.grid.rows} grid of ${f.tilesExamined} tiles; most likely ${f.tilesWithSubject} of them hold the subject, and between ${f.tilesLow} and ${f.tilesHigh} with 90% probability`;

const contextLines = (f: FactSheet) =>
  Object.fromEntries(
    Object.entries(f.context).map(([k, v]) => [
      k,
      v === null ? "unknown — the detector abstained" : v.toFixed(2),
    ]),
  );

export async function POST(request: Request) {
  let question: string;
  let facts: FactSheet;
  let reading: Reading;
  try {
    ({ question, facts, reading } = await request.json());
  } catch {
    return NextResponse.json({ error: "Body must be JSON." }, { status: 400 });
  }
  if (!facts?.kind) {
    return NextResponse.json({ error: "No fact sheet to judge." }, { status: 400 });
  }

  const asked = READINGS[reading] ?? READINGS.only;

  try {
    switch (facts.kind) {
      // ------------------------------------------------------------ identify
      case "identify": {
        const candidates = facts.tallies.map((t) => t.label);
        if (facts.wholeImage && !candidates.includes(facts.wholeImage.label)) {
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
         * choice trivial and Jev reports confidence 1.00 for a label backed by
         * one shaky observation.
         */
        const criteria: Record<string, string | null> = Object.fromEntries(
          candidates.map((label) => [label, null]),
        );
        criteria[UNDECIDED] =
          "The observations are too thin, too few or too uncertain to name one kind.";

        const { answers, usage } = await jev().systemOne({
          state: {
            question,
            what_the_question_asks_for: asked,
            observations: {
              how_the_picture_was_examined: coverageLine(facts),
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
              other_observations: contextLines(facts),
            },
            caveat: CAVEAT,
          },
          questions: {
            answer: choice(
              "Given only these observations, which label best answers the question?",
              criteria,
            ),
            mixed: noul(
              "Do the observations show more than one kind covering a meaningful part of the picture?",
            ),
            support: score("How well do these observations support a single confident answer?", SUPPORT),
          },
        });

        return NextResponse.json({
          answer: answers.answer.choice,
          confidence: answers.answer.confidence,
          probabilities: answers.answer.probabilities as Record<string, number>,
          mixed: answers.mixed.noul,
          support: answers.support.score,
          supportLegend: answers.support.legend as Record<string, unknown>,
          inputTokens: usage.input_tokens,
          usdCost: usdCost(usage.input_tokens),
        } satisfies Judgment);
      }

      // ------------------------------------------------------------ presence
      case "presence": {
        const { answers, usage } = await jev().systemOne({
          state: {
            question,
            what_the_question_asks_for: asked,
            observations: {
              the_sentence_that_was_scored: facts.statement,
              how_well_the_whole_picture_matched_it:
                facts.probability === null
                  ? "unknown — the detector abstained, which means it could not tell"
                  : facts.probability.toFixed(2),
              how_the_picture_was_examined: coverageLine(facts),
              tiles_that_also_matched_the_subject: `${facts.tilesMatchingSubject} of the ${facts.tilesChecked} tiles worth checking (the other ${facts.tilesExamined - facts.tilesChecked} held nothing)`,
              other_observations: contextLines(facts),
            },
            caveat: CAVEAT,
          },
          questions: {
            present: noul(`Is there a ${facts.subject} in this picture?`, {
              true: "The observations show it is there.",
              false: "The observations show it is absent, or are too weak to say it is there.",
            }),
            support: score("How well do these observations support a confident verdict?", SUPPORT),
          },
        });

        return NextResponse.json({
          answer: answers.present.noul >= 0.5 ? `yes — ${facts.subject}` : `no — no ${facts.subject}`,
          confidence: Math.max(answers.present.noul, 1 - answers.present.noul),
          present: answers.present.noul,
          support: answers.support.score,
          supportLegend: answers.support.legend as Record<string, unknown>,
          inputTokens: usage.input_tokens,
          usdCost: usdCost(usage.input_tokens),
        } satisfies Judgment);
      }

      // --------------------------------------------------------------- count
      case "count": {
        const { answers, usage } = await jev().systemOne({
          state: {
            question,
            what_the_question_asks_for: asked,
            observations: {
              what_was_counted: `tiles of the picture holding ${facts.noun}s`,
              how_the_picture_was_examined: coverageLine(facts),
              other_observations: contextLines(facts),
            },
            caveat: CAVEAT,
          },
          questions: {
            dependable: noul(
              "Do these observations give a count worth reporting, rather than a guess?",
            ),
            support: score("How well do these observations support a confident count?", SUPPORT),
          },
        });

        const spread =
          facts.tilesLow === facts.tilesHigh
            ? `${facts.tilesWithSubject} of ${facts.tilesExamined} tiles`
            : `${facts.tilesWithSubject} of ${facts.tilesExamined} tiles, between ${facts.tilesLow} and ${facts.tilesHigh}`;

        return NextResponse.json({
          answer: spread,
          confidence: answers.dependable.noul,
          present: answers.dependable.noul,
          support: answers.support.score,
          supportLegend: answers.support.legend as Record<string, unknown>,
          inputTokens: usage.input_tokens,
          usdCost: usdCost(usage.input_tokens),
        } satisfies Judgment);
      }

      // -------------------------------------------------------------- rating
      case "rating": {
        const scale = SCALES[facts.scale];
        const rubric = scale.levels as unknown as readonly [string, string, ...string[]];
        const { answers, usage } = await jev().systemOne({
          state: {
            question,
            what_the_question_asks_for: asked,
            observations: {
              the_scale_that_was_read: scale.description,
              where_the_picture_landed_on_it: `${facts.position.toFixed(2)} on a scale from 0 (${scale.levels[0]}) to 1 (${scale.levels[scale.levels.length - 1]})`,
              how_reliable_that_reading_was:
                facts.confidence === null
                  ? "unknown — the reading was too uncertain to report"
                  : facts.confidence.toFixed(2),
              other_observations: contextLines(facts),
            },
            caveat: CAVEAT,
          },
          questions: {
            level: score(
              "Given these observations, where does the picture sit on this scale?",
              rubric,
            ),
            support: score("How well do these observations support a confident rating?", SUPPORT),
          },
        });

        const nearest = Math.min(
          Math.max(Math.round(answers.level.score), 0),
          scale.levels.length - 1,
        );
        return NextResponse.json({
          answer: scale.levels[nearest],
          confidence: answers.level.confidence,
          support: answers.support.score,
          supportLegend: answers.support.legend as Record<string, unknown>,
          inputTokens: usage.input_tokens,
          usdCost: usdCost(usage.input_tokens),
        } satisfies Judgment);
      }
    }
  } catch (error) {
    const { message, status } = jevError(error);
    return NextResponse.json({ error: message }, { status });
  }
}
