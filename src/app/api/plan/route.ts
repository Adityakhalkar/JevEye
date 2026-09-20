/**
 * Jev decides what the question means, before any pixel is touched.
 *
 * Four judgments in one request, because System One reads the state once and
 * answers every question in parallel: is this even a question about the image,
 * what shape of answer does it want, which shipped label set is it about, and
 * which shipped scale — if any — is it asking along.
 *
 * The reading it returns decides which probes the vision layer will run, so
 * this call is what makes the question steer the looking.
 */
import { choice, noul } from "@typesafe-ai/sdk";
import { NextResponse } from "next/server";

import { jev, jevError, usdCost } from "@/lib/jev";
import { SCALES, type ScaleId } from "@/lib/scales";
import { READINGS, type Plan, type Reading } from "@/lib/types";
import { VOCABULARIES, type VocabularyId } from "@/lib/vocab";

/** Below this, the question is not about the picture and no probes run. */
const ON_TOPIC_GATE = 0.5;

const NO_SCALE = "none of these";

export async function POST(request: Request) {
  let question: string;
  try {
    ({ question } = await request.json());
  } catch {
    return NextResponse.json({ error: "Body must be JSON." }, { status: 400 });
  }
  if (typeof question !== "string" || question.trim() === "") {
    return NextResponse.json({ error: "Ask a question." }, { status: 400 });
  }

  const vocabularyCriteria = Object.fromEntries(
    Object.values(VOCABULARIES).map((v) => [v.id, v.description]),
  ) as Record<VocabularyId, string>;

  const scaleCriteria: Record<string, string> = {
    ...Object.fromEntries(Object.values(SCALES).map((s) => [s.id, s.description])),
    [NO_SCALE]: "The question is not asking how much of some quality the picture shows.",
  };

  try {
    const { answers, usage } = await jev().systemOne({
      state: { question },
      questions: {
        on_topic: noul(
          "Could this question be answered by looking at a photograph the asker just supplied?",
          {
            true: "It asks about what is visible in the picture.",
            false:
              "It asks about something else — general knowledge, the weather, arithmetic, or the software itself.",
          },
        ),
        reading: choice("What is the question actually asking for?", READINGS),
        vocabulary: choice(
          "Which catalogue of labels would be needed to answer this question?",
          vocabularyCriteria,
        ),
        scale: choice("Which scale, if any, is this question asking along?", scaleCriteria),
      },
    });

    const vocabulary = answers.vocabulary.choice as VocabularyId;
    const reading = answers.reading.choice as Reading;
    const scaleChoice = answers.scale.choice;
    const scale = scaleChoice === NO_SCALE ? null : (scaleChoice as ScaleId);
    let tokens = usage.input_tokens;

    /**
     * Checking for one particular thing needs to know which thing. Jev picks it
     * from the vocabulary — a second call, because the label set to choose from
     * is only known once the first call has settled the vocabulary.
     */
    let subject: string | null = null;
    let subjectConfidence: number | null = null;
    if (reading === "presence") {
      const labels = VOCABULARIES[vocabulary].labels;
      const picked = await jev().systemOne({
        state: { question },
        questions: {
          subject: choice(
            "Which one of these is the question asking about?",
            Object.fromEntries(labels.map((l) => [l, null])),
          ),
        },
      });
      subject = picked.answers.subject.choice;
      subjectConfidence = picked.answers.subject.confidence;
      tokens += picked.usage.input_tokens;
    }

    // A rating question with no scale to read has nothing to measure; fall back
    // to naming what is there rather than inventing a scale, which Jev cannot do.
    const effectiveReading: Reading = reading === "rating" && scale === null ? "only" : reading;

    const plan: Plan = {
      onTopic: answers.on_topic.noul,
      proceed: answers.on_topic.noul >= ON_TOPIC_GATE,
      reading: effectiveReading,
      readingConfidence: answers.reading.confidence,
      vocabulary,
      vocabularyConfidence: answers.vocabulary.confidence,
      instanceNoun: VOCABULARIES[vocabulary].instanceNoun,
      subject,
      subjectConfidence,
      // Only a rating reading reads a scale, so do not report one otherwise:
      // the plan should say what will actually happen, not what was considered.
      scale: effectiveReading === "rating" ? scale : null,
      scaleConfidence:
        effectiveReading === "rating" && scale !== null ? answers.scale.confidence : null,
      inputTokens: tokens,
      usdCost: usdCost(tokens),
    };
    return NextResponse.json(plan);
  } catch (error) {
    const { message, status } = jevError(error);
    return NextResponse.json({ error: message }, { status });
  }
}
