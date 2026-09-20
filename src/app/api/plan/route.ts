/**
 * Jev decides what the question means, before any pixel is touched.
 *
 * Three judgments in one request, because System One reads the state once and
 * answers every question in parallel: is this even a question about the image,
 * which shipped label set is it about, and what shape of answer does it want.
 */
import { choice, noul } from "@typesafe-ai/sdk";
import { NextResponse } from "next/server";

import { jev, jevError, usdCost } from "@/lib/jev";
import { READINGS, type Plan, type Reading } from "@/lib/types";
import { VOCABULARIES, type VocabularyId } from "@/lib/vocab";

/** Below this, the question is not about the picture and no probes run. */
const ON_TOPIC_GATE = 0.5;

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
        vocabulary: choice(
          "Which catalogue of labels would be needed to answer this question?",
          vocabularyCriteria,
        ),
        reading: choice("What is the question actually asking for?", READINGS),
      },
    });

    const vocabulary = answers.vocabulary.choice as VocabularyId;
    const plan: Plan = {
      onTopic: answers.on_topic.noul,
      proceed: answers.on_topic.noul >= ON_TOPIC_GATE,
      vocabulary,
      vocabularyConfidence: answers.vocabulary.confidence,
      reading: answers.reading.choice as Reading,
      readingConfidence: answers.reading.confidence,
      instanceNoun: VOCABULARIES[vocabulary].instanceNoun,
      inputTokens: usage.input_tokens,
      usdCost: usdCost(usage.input_tokens),
    };
    return NextResponse.json(plan);
  } catch (error) {
    const { message, status } = jevError(error);
    return NextResponse.json({ error: message }, { status });
  }
}
