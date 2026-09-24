"use client";

import Link from "next/link";
import { useCallback, useRef, useState } from "react";

import { CALIBRATION } from "@/lib/calibration";
import type { FactSheet, Judgment, Plan } from "@/lib/types";
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
        <p className="mt-1 max-w-xl text-neutral-400">
          A CNN reports what it sees, with a calibrated confidence or an abstention. Jev judges what
          that means. The image never leaves this browser — only the fact sheet does.{" "}
          <Link href="/live" className="underline hover:text-neutral-300">
            live video →
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
          // eslint-disable-next-line @next/next/no-img-element
          <img src={preview} alt="the image being asked about" className="max-h-72 w-auto rounded" />
        ) : (
          <p className="py-8 text-center text-neutral-500">drop an image here, or click to choose</p>
        )}
      </section>

      <section className="flex gap-2">
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !busy && ask()}
          placeholder="ask about the image…"
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

      {plan && (
        <Block title="jev reads the question">
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
        <Block title="not about the image">
          <p className="text-neutral-300">
            Jev puts the odds that this question is about the picture at{" "}
            {plan?.onTopic.toFixed(2)}. No probes ran and nothing was spent on judging.
          </p>
        </Block>
      )}

      {download && (
        <Block title="loading models (first visit only)">
          <Row k={download.file} v={`${download.percent}%`} />
        </Block>
      )}

      {stages.length > 0 && (
        <Block title={`vision probes${facts ? ` · ${facts.elapsedMs} ms, on this machine` : ""}`}>
          {stages.map((s, i) => (
            <div key={i} className="text-neutral-400">
              {i === stages.length - 1 && busy ? "▸ " : "  "}
              {s}
            </div>
          ))}
        </Block>
      )}

      {judgment && facts && (
        <Block title="answer">
          <p className="text-base text-neutral-50">
            {judgment.answer}
            <span className="ml-2 text-sm text-neutral-500">
              confidence {judgment.confidence.toFixed(2)}
            </span>
          </p>
          <p className="mt-2 text-neutral-300">{explain(facts, judgment)}</p>
          <Row k="support" v={judgment.support.toFixed(2)} />
          <Row k="cost" v={`$${judgment.usdCost.toFixed(6)} · ${judgment.inputTokens} tokens`} />
        </Block>
      )}

      {facts && (
        <details className="rounded-lg border border-neutral-800 bg-neutral-900/40">
          <summary className="cursor-pointer px-4 py-3 text-neutral-300">
            what I saw ({facts.kind})
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
        <Block title="error">
          <p className="text-red-400">{error}</p>
        </Block>
      )}
    </main>
  );
}

/** One plain sentence about what the evidence was, in the reading's own terms. */
function explain(facts: FactSheet, judgment: Judgment): string {
  switch (facts.kind) {
    case "identify": {
      const spread =
        facts.tilesLow === facts.tilesHigh
          ? `Found in ${facts.tilesWithSubject} of ${facts.tilesExamined} tiles.`
          : `Found in most likely ${facts.tilesWithSubject} of ${facts.tilesExamined} tiles, between ${facts.tilesLow} and ${facts.tilesHigh}.`;
      const mixed =
        judgment.mixed === undefined
          ? ""
          : judgment.mixed > 0.5
            ? ` More than one kind covers a meaningful part of the picture (${judgment.mixed.toFixed(2)}), so this is not a single-kind image.`
            : ` Jev puts the odds of a mixed picture at ${judgment.mixed.toFixed(2)}.`;
      return spread + mixed;
    }
    case "presence": {
      const seen = facts.topLabels[0];
      const detector =
        facts.detected === null
          ? `The detector does not know "${facts.subject}", so this rests on the labels alone.`
          : facts.detectorScore !== null
            ? `The detector found ${facts.subject} at ${facts.detectorScore.toFixed(2)}.`
            : `The detector searched for ${facts.subject} and localised none.`;
      return `${detector} With every label competing, "${facts.subject}" took ${facts.subjectProbability.toFixed(3)} and placed ${facts.subjectRank}; the strongest was ${seen ? `"${seen.label}" at ${seen.p.toFixed(3)}` : "none"}.`;
    }
    case "count":
      return facts.instances
        ? `The detector localised them one by one: most likely ${facts.instances.mode}, between ${facts.instances.low} and ${facts.instances.high} with 90% probability.`
        : `The detector does not know this category, so this counts tiles rather than individuals: ${facts.tilesWithSubject} of ${facts.tilesExamined} hold ${facts.noun}s.`;
    case "rating":
      return `Read along the "${facts.scale}" scale, the picture landed at ${facts.position.toFixed(2)} of 1.00, ${
        facts.confidence === null
          ? "but the reading was too uncertain to report a reliability"
          : `with reading reliability ${facts.confidence.toFixed(2)}`
      }.`;
  }
}

function Block({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="rounded-lg border border-neutral-800 bg-neutral-900/40 px-4 py-3">
      <h2 className="mb-2 text-xs uppercase tracking-widest text-neutral-500">{title}</h2>
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
