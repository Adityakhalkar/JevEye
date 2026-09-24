"use client";

import Link from "next/link";
import { useCallback, useRef, useState } from "react";

import { Boxes, Toggle } from "@/components/Boxes";
import { CALIBRATION } from "@/lib/calibration";
import { detectionsOf, type FactSheet, type Judgment, type Plan } from "@/lib/types";
import { probe, prepare, warmUp, type LoadProgress } from "@/lib/vision";
import { VOCABULARIES } from "@/lib/vocab";

type Status = "idle" | "loading" | "planning" | "probing" | "judging" | "done" | "offtopic" | "error";

const EXAMPLES = [
  "what type of flower is it?",
  "is there a dog in this photo?",
  "how much of this is covered in flowers?",
  "do these plants look healthy?",
  "is this photo blurry?",
];

export default function Page() {
  const [preview, setPreview] = useState<string | null>(null);
  const [file, setFile] = useState<Blob | null>(null);
  const [question, setQuestion] = useState("");
  const [status, setStatus] = useState<Status>("idle");
  const [stages, setStages] = useState<string[]>([]);
  const [download, setDownload] = useState<LoadProgress | null>(null);
  const [plan, setPlan] = useState<Plan | null>(null);
  const [facts, setFacts] = useState<FactSheet | null>(null);
  const [judgment, setJudgment] = useState<Judgment | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showBoxes, setShowBoxes] = useState(true);
  const [showNumbers, setShowNumbers] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  const accept = useCallback((chosen: File | null | undefined) => {
    if (!chosen || !chosen.type.startsWith("image/")) {
      setError("That is not an image this browser can decode. HEIC often fails; try JPEG or PNG.");
      return;
    }
    setFile(chosen);
    setPreview(URL.createObjectURL(chosen));
    setStatus("idle");
    setPlan(null);
    setFacts(null);
    setJudgment(null);
    setError(null);
    setStages([]);
  }, []);

  async function ask() {
    if (!file || question.trim() === "") return;
    setError(null);
    setPlan(null);
    setFacts(null);
    setJudgment(null);
    setStages([]);

    try {
      setStatus("planning");
      const planned = await fetch("/api/plan", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ question }),
      });
      const planJson = await planned.json();
      if (!planned.ok) throw new Error(planJson.error ?? "Planning failed.");
      const nextPlan = planJson as Plan;
      setPlan(nextPlan);

      if (!nextPlan.proceed) {
        setStatus("offtopic");
        return;
      }

      setStatus("loading");
      await warmUp(setDownload);
      setDownload(null);

      setStatus("probing");
      const image = await prepare(file);
      const sheet = await probe(image, nextPlan, (stage) =>
        setStages((prior) => [...prior, stage]),
      );
      setFacts(sheet);

      setStatus("judging");
      const judged = await fetch("/api/judge", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ question, facts: sheet, reading: nextPlan.reading }),
      });
      const judgeJson = await judged.json();
      if (!judged.ok) throw new Error(judgeJson.error ?? "Judging failed.");
      setJudgment(judgeJson as Judgment);
      setStatus("done");
    } catch (thrown) {
      setError(thrown instanceof Error ? thrown.message : String(thrown));
      setStatus("error");
    }
  }

  const busy = ["loading", "planning", "probing", "judging"].includes(status);

  return (
    <main className="mx-auto flex min-h-screen max-w-3xl flex-col gap-6 px-4 py-10 font-mono text-sm">
      <header>
        <h1 className="text-lg font-semibold tracking-tight text-neutral-50">JevEye</h1>
        <p className="mt-1 max-w-[62ch] text-neutral-400">
          Drop in a picture and ask about it. One model finds and names what is there, another works
          out what that means for your question — and both say how sure they are. Your picture stays
          on this device.{" "}
          <Link href="/live" className="text-neutral-300 underline underline-offset-2 hover:text-neutral-100">
            Watch live video instead
          </Link>
        </p>
      </header>

      <section
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          accept(e.dataTransfer.files?.[0]);
        }}
        onClick={() => input.current?.click()}
        className="cursor-pointer rounded-lg border border-dashed border-neutral-700 p-4 transition hover:border-neutral-500"
      >
        <input
          ref={input}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => accept(e.target.files?.[0])}
        />
        {preview ? (
          <span className="relative inline-block">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={preview} alt="the picture being asked about" className="block max-h-72 w-auto rounded" />
            {showBoxes && facts && detectionsOf(facts).length > 0 && (
              <Boxes found={detectionsOf(facts)} size={facts.imageSize} />
            )}
          </span>
        ) : (
          <p className="py-8 text-center text-neutral-500">
            Drop a picture here, or click to choose one
          </p>
        )}
      </section>

      {facts && detectionsOf(facts).length > 0 && (
        <Toggle checked={showBoxes} onChange={setShowBoxes}>
          Outline what was found ({detectionsOf(facts).length})
        </Toggle>
      )}

      <section className="flex gap-2">
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !busy && ask()}
          placeholder="Ask about this picture…"
          className="flex-1 rounded border border-neutral-700 bg-neutral-900 px-3 py-2 outline-none placeholder:text-neutral-600 focus:border-neutral-500"
        />
        <button
          onClick={ask}
          disabled={busy || !file || question.trim() === ""}
          className="rounded bg-neutral-100 px-4 py-2 font-semibold text-neutral-900 disabled:cursor-not-allowed disabled:bg-neutral-800 disabled:text-neutral-500"
        >
          {busy ? "…" : "ask"}
        </button>
      </section>

      {!preview && (
        <div className="flex flex-wrap gap-2 text-xs text-neutral-500">
          {EXAMPLES.map((e) => (
            <button
              key={e}
              onClick={() => setQuestion(e)}
              className="rounded border border-neutral-800 px-2 py-1 hover:border-neutral-600"
            >
              {e}
            </button>
          ))}
        </div>
      )}

      {plan && showNumbers && (
        <Block title="how the question was read">
          <Row k="on topic" v={plan.onTopic.toFixed(2)} />
          <Row k="reading" v={`${plan.reading} (${plan.readingConfidence.toFixed(2)})`} />
          {plan.reading === "rating" && plan.scale ? (
            <Row k="scale" v={`${plan.scale} (${plan.scaleConfidence?.toFixed(2)})`} />
          ) : (
            <Row
              k="vocabulary"
              v={`${plan.vocabulary} (${plan.vocabularyConfidence.toFixed(2)}) — ${VOCABULARIES[plan.vocabulary].labels.length} labels`}
            />
          )}
          {plan.subject && (
            <Row k="subject" v={`${plan.subject} (${plan.subjectConfidence?.toFixed(2)})`} />
          )}
          <Row k="cost" v={`$${plan.usdCost.toFixed(6)} · ${plan.inputTokens} tokens`} />
        </Block>
      )}

      {status === "offtopic" && (
        <Block title="Not about the picture">
          <p className="text-neutral-300">
            Jev puts the odds that this question is about the picture at{" "}
            {plan?.onTopic.toFixed(2)}. No probes ran and nothing was spent on judging.
          </p>
        </Block>
      )}

      {download && (
        <Block title="Getting ready — this happens once">
          <Row k={download.file} v={`${download.percent}%`} />
        </Block>
      )}

      {stages.length > 0 && (busy || showNumbers) && (
        <Block title={`looking${facts ? `, took ${(facts.elapsedMs / 1000).toFixed(1)}s on this device` : ""}`}>
          {stages.map((s, i) => (
            <div key={i} className="text-neutral-400">
              {i === stages.length - 1 && busy ? "▸ " : "  "}
              {s}
            </div>
          ))}
        </Block>
      )}

      {judgment && facts && (
        <Block title="Answer">
          <p className="text-base text-neutral-50">
            {judgment.answer}
            {facts.kind !== "count" && (
              <span className="ml-2 text-sm text-neutral-500">
                {describeConfidence(judgment.confidence)}
              </span>
            )}
          </p>
          <p className="mt-2 text-neutral-300">{explain(facts, judgment)}</p>
          {showNumbers && (
            <>
              <Row k="how well the evidence supports it, 0 to 2" v={judgment.support.toFixed(2)} />
              <Row k="cost of this answer" v={`$${judgment.usdCost.toFixed(6)}`} />
            </>
          )}
        </Block>
      )}

      {facts && (
        <Toggle checked={showNumbers} onChange={setShowNumbers}>
          Show the numbers behind this
        </Toggle>
      )}

      {facts && showNumbers && (
        <details open className="rounded-lg border border-neutral-800 bg-neutral-900/40">
          <summary className="cursor-pointer px-4 py-3 text-neutral-300">
            Everything it measured
          </summary>
          <div className="space-y-1 border-t border-neutral-800 px-4 py-3">
            {facts.kind !== "rating" && (
              <>
                <Row
                  k="grid"
                  v={`${facts.grid.cols}×${facts.grid.rows} — ${facts.tilesExamined} tiles`}
                />
                <Row
                  k="tiles holding the subject"
                  v={`${facts.tilesWithSubject} (90% between ${facts.tilesLow} and ${facts.tilesHigh})`}
                />
              </>
            )}

            {facts.kind === "identify" && (
              <>
                <Row
                  k="named by"
                  v={
                    facts.classifier === "detector"
                      ? "the detector, which localises things rather than guessing what the frame resembles"
                      : facts.classifier === "probe" && facts.classifierAccuracy !== null
                        ? `a trained probe — ${(facts.classifierAccuracy * 100).toFixed(1)}% on held-out images, ECE ${facts.classifierEce?.toFixed(3)}`
                        : "zero-shot text, uncalibrated"
                  }
                />
                {facts.tallies.map((t) => (
                  <Row
                    key={t.label}
                    k={t.label}
                    v={`${t.count} tiles · mean ${t.meanConfidence.toFixed(2)} · weight ${t.weight.toFixed(2)}`}
                  />
                ))}
                {facts.unknownTiles > 0 && (
                  <Row k="tiles that could not be named" v={String(facts.unknownTiles)} />
                )}
                {facts.wholeImage && (
                  <Row
                    k="whole-image reading"
                    v={`${facts.wholeImage.label} · ${facts.wholeImage.confidence.toFixed(2)}`}
                  />
                )}
              </>
            )}

            {facts.kind === "presence" && (
              <>
                <Row
                  k="classifier"
                  v={facts.classifier === "probe" ? "trained probe" : "zero-shot text"}
                />
                <Row
                  k={`detector looked for "${facts.subject}"`}
                  v={
                    facts.detected === null
                      ? "not consulted — outside its 80 categories"
                      : facts.detectorScore === null
                        ? "found none"
                        : `found it at ${facts.detectorScore.toFixed(2)}`
                  }
                />
                {facts.detected?.slice(0, 3).map((d, i) => (
                  <Row
                    key={`${d.label}-${i}`}
                    k={`detector saw ${d.label}`}
                    v={`${d.score.toFixed(2)} · ${Math.round(d.area * 100)}% of frame`}
                  />
                ))}
                <Row
                  k={`labels ranked "${facts.subject}"`}
                  v={`${facts.subjectProbability.toFixed(3)} · rank ${facts.subjectRank}`}
                />
                {facts.topLabels.map((t) => (
                  <Row key={t.label} k={`sees ${t.label}`} v={t.p.toFixed(3)} />
                ))}
                <Row
                  k="tiles choosing the subject"
                  v={`${facts.tilesMatchingSubject} of ${facts.tilesChecked} checked (${facts.tilesExamined - facts.tilesChecked} held nothing)`}
                />
              </>
            )}

            {facts.kind === "count" && (
              <>
                <Row
                  k="how many are there"
                  v={
                    facts.instances
                      ? `${facts.instances.mode} (90% between ${facts.instances.low} and ${facts.instances.high}), counted by the detector`
                      : `not counted — outside the detector's 80 categories, so the tiles above measure coverage instead`
                  }
                />
                {facts.detected?.map((d, i) => (
                  <Row
                    key={`${d.label}-${i}`}
                    k={`found ${d.label}`}
                    v={`${d.score.toFixed(2)} · ${Math.round(d.area * 100)}% of frame`}
                  />
                ))}
              </>
            )}

            {facts.kind === "rating" && (
              <>
                <Row k="scale" v={facts.scale} />
                <Row k="position" v={`${facts.position.toFixed(2)} of 1.00`} />
                <Row
                  k="reading reliability"
                  v={facts.confidence === null ? "unknown — abstained" : facts.confidence.toFixed(2)}
                />
                {facts.levels.map((l, i) => (
                  <Row key={l} k={`level ${i}`} v={l} />
                ))}
              </>
            )}

            {Object.entries(facts.context).map(([k, v]) => (
              <Row key={k} k={k} v={v === null ? "unknown — abstained" : v.toFixed(2)} />
            ))}
            <Row k="image" v={`${facts.imageSize.width}×${facts.imageSize.height}`} />
            <Row k="vision time" v={`${facts.elapsedMs} ms, on this machine`} />
            {facts.kind === "identify" && facts.classifier !== "zero-shot" ? (
              <p className="pt-2 text-xs text-neutral-500">
                {facts.kind === "identify" && facts.classifier === "detector"
                  ? "The naming above comes from a detector that localises objects, not from a guess about the whole frame. Its scores are its own and are not calibrated here."
                  : "The naming above comes from a probe fitted on Oxford Flowers-102 and calibrated on images it never saw."}{" "}
                The coverage and context numbers on this page are still raw model outputs, and are
                probably too confident.
              </p>
            ) : (
              !CALIBRATION.fitted && (
                <p className="pt-2 text-xs text-amber-500/80">
                  Calibration is unfitted: these confidences are raw model outputs and are probably
                  too high.
                </p>
              )
            )}
          </div>
        </details>
      )}

      {error && (
        <Block title="Something went wrong">
          <p className="text-red-400">{error}</p>
        </Block>
      )}
    </main>
  );
}

/**
 * Confidence in words as well as a number.
 *
 * "0.39" tells a visitor nothing about whether to believe the answer, and the
 * number alone reads as precision the calibration does not yet earn outside the
 * flowers probe.
 */
function describeConfidence(p: number): string {
  const word = p >= 0.85 ? "very sure" : p >= 0.6 ? "fairly sure" : p >= 0.4 ? "unsure" : "guessing";
  return `${word} · ${p.toFixed(2)}`;
}

/** One plain sentence about what the evidence was, in the reading's own terms. */
function explain(facts: FactSheet, judgment: Judgment): string {
  switch (facts.kind) {
    case "identify": {
      const boxes = facts.detected ?? [];
      const spread =
        facts.classifier === "detector" && boxes.length > 0
          ? `It picked out ${boxes.length === 1 ? "one thing" : `${boxes.length} things`}: ${boxes
              .map((b) => `${b.label} at ${Math.round(b.score * 100)}%`)
              .join(", ")}.`
          : facts.tilesLow === facts.tilesHigh
            ? `It checked the picture in ${facts.tilesExamined} patches and found this in ${facts.tilesWithSubject} of them.`
            : `It checked the picture in ${facts.tilesExamined} patches and found this in about ${facts.tilesWithSubject} of them.`;
      const mixed =
        judgment.mixed === undefined
          ? ""
          : judgment.mixed > 0.5
            ? " There is clearly more than one kind here, so a single answer does not cover it."
            : " It reads as one kind rather than a mixture.";
      return spread + mixed;
    }
    case "presence": {
      const seen = facts.topLabels[0];
      const detector =
        facts.detected === null
          ? `It cannot look for "${facts.subject}" directly, so this rests on what the picture most resembles.`
          : facts.detectorScore !== null
            ? `It found ${facts.subject} at ${Math.round(facts.detectorScore * 100)}%.`
            : `It looked for ${facts.subject} and found none anywhere in the picture.`;
      const alternative = seen ? ` The closest match was "${seen.label}".` : "";
      return `${detector}${alternative}`;
    }
    case "count":
      return facts.instances
        ? facts.instances.low === facts.instances.high
          ? `It picked them out one by one and is settled on ${facts.instances.mode}.`
          : `It picked them out one by one — most likely ${facts.instances.mode}, and somewhere between ${facts.instances.low} and ${facts.instances.high}.`
        : `It cannot pick these out individually, so this is how much of the picture they fill rather than how many there are: ${facts.tilesWithSubject} patches of ${facts.tilesExamined}.`;
    case "rating":
      return `Judged on ${facts.scale}, the picture sits ${Math.round(facts.position * 100)}% of the way up the scale${
        facts.confidence === null ? ", though it was not confident about that reading" : ""
      }.`;
  }
}

function Block({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="rounded-lg border border-neutral-800 bg-neutral-900/40 px-4 py-3">
      <h2 className="mb-2 text-neutral-500">{title}</h2>
      {children}
    </section>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex justify-between gap-4">
      <span className="text-neutral-500">{k}</span>
      <span className="text-right text-neutral-200">{v}</span>
    </div>
  );
}
