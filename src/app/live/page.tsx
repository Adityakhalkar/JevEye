"use client";

import { RawImage } from "@huggingface/transformers";
import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";

import {
  CHANGE_THRESHOLD,
  LiveWindow,
  toSample,
  type LiveFacts,
} from "@/lib/live";
import { classify, embedImages, similarity, warmUp, type LoadProgress } from "@/lib/vision";
import { VOCABULARIES } from "@/lib/vocab";

/** Sampling rate. Embedding costs ~15ms, so this leaves the tab responsive. */
const SAMPLE_MS = 300;
const FRAME_SIZE = 336;

type Judgment = {
  situation: string;
  confidence: number;
  attention: number;
  settled: number;
  usdCost: number;
  inputTokens: number;
  at: number;
};

type Entry = Judgment & { reason: string };

export default function LivePage() {
  const video = useRef<HTMLVideoElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const window_ = useRef(new LiveWindow());
  const running = useRef(false);
  const inFlight = useRef(false);

  const [status, setStatus] = useState("idle");
  const [download, setDownload] = useState<LoadProgress | null>(null);
  const [facts, setFacts] = useState<LiveFacts | null>(null);
  const [entries, setEntries] = useState<Entry[]>([]);
  const [watching, setWatching] = useState("");
  const [spend, setSpend] = useState(0);
  const [error, setError] = useState<string | null>(null);
  // The sampling loop is long-lived, so it reads this through a ref rather than
  // being torn down and restarted every time the text changes.
  const watchingRef = useRef(watching);
  useEffect(() => {
    watchingRef.current = watching;
  }, [watching]);

  const judge = useCallback(async (live: LiveFacts, reason: string) => {
    if (inFlight.current) return;
    inFlight.current = true;
    try {
      const response = await fetch("/api/live-judge", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ facts: live, watching: watchingRef.current }),
      });
      const json = await response.json();
      if (!response.ok) throw new Error(json.error ?? "Judging failed.");
      setEntries((prior) => [{ ...(json as Judgment), reason }, ...prior].slice(0, 12));
      setSpend((s) => s + (json.usdCost ?? 0));
    } catch (thrown) {
      setError(thrown instanceof Error ? thrown.message : String(thrown));
    } finally {
      inFlight.current = false;
    }
  }, []);

  /** One turn of the fast loop: grab, embed, name, then decide if Jev should look. */
  const tick = useCallback(async () => {
    const element = video.current;
    const surface = canvas.current;
    if (!element || !surface || element.readyState < 2) return;

    const ratio = element.videoHeight / element.videoWidth || 1;
    surface.width = FRAME_SIZE;
    surface.height = Math.round(FRAME_SIZE * ratio);
    const context = surface.getContext("2d");
    if (!context) return;
    context.drawImage(element, 0, 0, surface.width, surface.height);

    const frame = RawImage.fromCanvas(surface);
    const [embed] = await embedImages([frame]);
    const [named] = await classify([embed], VOCABULARIES.objects);

    const now = Date.now();
    window_.current.push(toSample(now, embed, named));

    const live = window_.current.facts(now, similarity);
    setFacts(live);

    const { judge: should, reason } = window_.current.shouldJudge(now, similarity);
    if (should) {
      window_.current.markJudged(now);
      void judge(live, reason);
    }
  }, [judge]);

  useEffect(() => {
    if (status !== "running") return;
    let cancelled = false;
    const loop = async () => {
      while (!cancelled && running.current) {
        const started = Date.now();
        try {
          await tick();
        } catch (thrown) {
          setError(thrown instanceof Error ? thrown.message : String(thrown));
          break;
        }
        const spent = Date.now() - started;
        await new Promise((r) => setTimeout(r, Math.max(0, SAMPLE_MS - spent)));
      }
    };
    void loop();
    return () => {
      cancelled = true;
    };
  }, [status, tick]);

  async function start(source: "camera" | MediaProvider | string) {
    setError(null);
    setStatus("loading");
    await warmUp(setDownload);
    setDownload(null);

    const element = video.current;
    if (!element) return;
    try {
      if (source === "camera") {
        element.srcObject = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480 },
          audio: false,
        });
      } else {
        element.srcObject = null;
        element.src = source as string;
        element.loop = true;
      }
      await element.play();
    } catch (thrown) {
      setError(
        thrown instanceof Error
          ? `Could not start the video: ${thrown.message}`
          : String(thrown),
      );
      setStatus("idle");
      return;
    }
    running.current = true;
    setStatus("running");
  }

  function stop() {
    running.current = false;
    const element = video.current;
    const stream = element?.srcObject as MediaStream | null;
    stream?.getTracks().forEach((t) => t.stop());
    if (element) element.srcObject = null;
    setStatus("idle");
  }

  const latest = entries[0];

  return (
    <main className="mx-auto flex min-h-screen max-w-3xl flex-col gap-6 px-4 py-10 font-mono text-sm">
      <header>
        <h1 className="text-lg font-semibold tracking-tight text-neutral-50">JevEye Live</h1>
        <p className="mt-1 max-w-xl text-neutral-400">
          Frames are embedded about every {SAMPLE_MS} ms and never leave this browser. Jev is asked
          only when the view actually moves — a cosine distance of {CHANGE_THRESHOLD} between
          embeddings is the gate, and it costs nothing to compute.
        </p>
      </header>

      <video ref={video} playsInline muted className="w-full rounded-lg border border-neutral-800" />
      <canvas ref={canvas} className="hidden" />

      <section className="flex flex-wrap gap-2">
        {status === "running" ? (
          <button onClick={stop} className="rounded bg-neutral-100 px-4 py-2 font-semibold text-neutral-900">
            stop
          </button>
        ) : (
          <>
            <button
              onClick={() => start("camera")}
              className="rounded bg-neutral-100 px-4 py-2 font-semibold text-neutral-900"
            >
              use camera
            </button>
            <button
              onClick={() => start("/sample.mp4")}
              className="rounded border border-neutral-700 px-4 py-2 text-neutral-300 hover:border-neutral-500"
            >
              use the sample clip
            </button>
          </>
        )}
        <input
          value={watching}
          onChange={(e) => setWatching(e.target.value)}
          placeholder="what should it watch for? (optional)"
          className="min-w-56 flex-1 rounded border border-neutral-700 bg-neutral-900 px-3 py-2 outline-none placeholder:text-neutral-600 focus:border-neutral-500"
        />
      </section>

      {download && (
        <Block title="loading models (first visit only)">
          <Row k={download.file} v={`${download.percent}%`} />
        </Block>
      )}

      {latest && (
        <Block title="what jev makes of it">
          <p className="text-base text-neutral-50">
            {latest.situation}
            <span className="ml-2 text-sm text-neutral-500">
              confidence {latest.confidence.toFixed(2)}
            </span>
          </p>
          <Row k="attention" v={latest.attention.toFixed(2)} />
          <Row k="settled" v={latest.settled.toFixed(2)} />
          <Row k="asked because" v={latest.reason} />
        </Block>
      )}

      {facts && (
        <Block
          title={`the window · ${facts.windowSeconds}s · ${facts.samples} samples at ${facts.samplesPerSecond}/s`}
        >
          {facts.subjects.length === 0 && <p className="text-neutral-500">nothing named yet</p>}
          {facts.subjects.map((s) => (
            <Row
              key={s.label}
              k={s.label}
              v={`${Math.round(s.seenFraction * 100)}% of samples · mean ${s.meanProbability.toFixed(2)} · ${s.trend} · last seen ${s.secondsSinceLastSeen.toFixed(1)}s ago`}
            />
          ))}
          <Row k="change since jev looked" v={facts.changeSinceLastJudgment.toFixed(3)} />
          <Row k="change between samples" v={facts.changeBetweenSamples.toFixed(3)} />
          <Row k="nothing nameable in" v={`${facts.unnamedSamples} of ${facts.samples}`} />
        </Block>
      )}

      {entries.length > 0 && (
        <Block title={`judgments · ${entries.length} shown · $${spend.toFixed(6)} total`}>
          {entries.map((e) => (
            <Row
              key={e.at}
              k={new Date(e.at).toLocaleTimeString()}
              v={`${e.situation} (${e.confidence.toFixed(2)}) — ${e.reason}`}
            />
          ))}
        </Block>
      )}

      {error && (
        <Block title="error">
          <p className="text-red-400">{error}</p>
        </Block>
      )}

      <p className="text-xs text-neutral-600">
        Judging is gated and rate-limited, so a still scene costs nothing. A frame is never
        uploaded: only the window summary, a few hundred tokens of text, crosses the network.{" "}
        <Link href="/" className="underline hover:text-neutral-400">
          still images →
        </Link>
      </p>
    </main>
  );
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
      <span className="shrink-0 text-neutral-500">{k}</span>
      <span className="text-right text-neutral-200">{v}</span>
    </div>
  );
}
