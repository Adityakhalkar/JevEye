/**
 * Which catalogue of labels a watcher means.
 *
 * A still-image question lets Jev pick the label set as part of reading the
 * question. A camera has no question, so the live view was pinned to COCO and
 * called a field of flowers "vase". The "watching for" box is the question it
 * was missing — one Jev call, asked once when the text settles rather than per
 * frame.
 */
import { choice } from "@typesafe-ai/sdk";
import { NextResponse } from "next/server";

import { jev, jevError, usdCost } from "@/lib/jev";
import { VOCABULARIES, type VocabularyId } from "@/lib/vocab";

export async function POST(request: Request) {
  let watching: string;
  try {
    ({ watching } = await request.json());
  } catch {
    return NextResponse.json({ error: "Body must be JSON." }, { status: 400 });
  }
  if (typeof watching !== "string" || watching.trim() === "") {
    return NextResponse.json({ error: "Nothing to interpret." }, { status: 400 });
  }

  const criteria = Object.fromEntries(
    Object.values(VOCABULARIES).map((v) => [v.id, v.description]),
  ) as Record<VocabularyId, string>;

  try {
    const { answers, usage } = await jev().systemOne({
      state: { watching_for: watching },
      questions: {
        vocabulary: choice(
          "Which catalogue of labels would be needed to recognise what this person is watching for?",
          criteria,
        ),
      },
    });
    return NextResponse.json({
      vocabulary: answers.vocabulary.choice as VocabularyId,
      confidence: answers.vocabulary.confidence,
      usdCost: usdCost(usage.input_tokens),
    });
  } catch (error) {
    const { message, status } = jevError(error);
    return NextResponse.json({ error: message }, { status });
  }
}
