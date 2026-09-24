/**
 * Jev judges a window of time rather than a frame.
 *
 * This is where a live view earns a semantic model at all. A per-frame
 * classifier can say a person is there; only a window can say they are getting
 * closer, or that the scene has been churning, or that nothing has settled.
 * Those are judgments about a trend, which is the shape Jev is good at.
 *
 * Situations are a shipped list, like vocabularies and scales: Jev picks one, it
 * cannot write one.
 */
import { choice, noul, score } from "@typesafe-ai/sdk";
import { NextResponse } from "next/server";

import { jev, jevError, usdCost } from "@/lib/jev";
import type { LiveFacts } from "@/lib/live";

export const SITUATIONS: Record<string, string> = {
  "something is arriving or coming closer":
    "A subject is present and taking up more of the view over the window, or has newly appeared.",
  "something is leaving or moving away":
    "A subject is taking up less of the view, or was there earlier in the window and is not now.",
  "the scene is steady":
    "Much the same throughout — whatever is there is there, without much movement.",
  "the scene changed to something different":
    "What the camera is looking at is not what it was looking at earlier in the window.",
  "cannot tell from these observations":
    "Too few samples, too much churn, or too little named to say anything.",
};

/** The window as Jev reads it: durations and directions, not pixels. */
function observations(facts: LiveFacts) {
  return {
    window: `the last ${facts.windowSeconds} seconds, sampled ${facts.samples} times at ${facts.samplesPerSecond} per second`,
    what_has_been_in_view: facts.subjects.map(
      (s) =>
        `${s.label}: led in ${Math.round(s.seenFraction * 100)}% of samples, mean confidence ${s.meanProbability.toFixed(2)}, ${s.trend} across the window, first seen ${s.secondsSinceFirstSeen.toFixed(1)}s ago and last seen ${s.secondsSinceLastSeen.toFixed(1)}s ago`,
    ),
    nothing_nameable_in: `${facts.unnamedSamples} of ${facts.samples} samples`,
    how_much_the_view_moved: `${facts.changeSinceLastJudgment} since you last looked, and ${facts.changeBetweenSamples} between consecutive samples on average (0 means an identical picture)`,
    how_many_different_things_led: facts.distinctLeaders,
  };
}

export async function POST(request: Request) {
  let facts: LiveFacts;
  let watching: string | undefined;
  try {
    ({ facts, watching } = await request.json());
  } catch {
    return NextResponse.json({ error: "Body must be JSON." }, { status: 400 });
  }
  if (!facts?.samples) {
    return NextResponse.json({ error: "No window to judge." }, { status: 400 });
  }

  try {
    const { answers, usage } = await jev().systemOne({
      state: {
        observations: observations(facts),
        what_the_watcher_cares_about: watching?.trim() || "anything worth noticing",
        caveat:
          "These come from an uncalibrated image model reading single frames, and a label leading a frame does not mean it is the only thing there. Trends matter more here than any single number.",
      },
      questions: {
        situation: choice("What is happening over this window?", SITUATIONS),
        attention: score("How much does this deserve someone's attention?", [
          "Not at all — nothing here worth a glance",
          "Worth noticing, but nothing needs doing",
          "Something has changed that someone should look at now",
        ]),
        settled: noul("Has the scene settled, rather than still changing?"),
      },
    });

    return NextResponse.json({
      situation: answers.situation.choice,
      confidence: answers.situation.confidence,
      attention: answers.attention.score,
      settled: answers.settled.noul,
      inputTokens: usage.input_tokens,
      usdCost: usdCost(usage.input_tokens),
      at: Date.now(),
    });
  } catch (error) {
    const { message, status } = jevError(error);
    return NextResponse.json({ error: message }, { status });
  }
}
