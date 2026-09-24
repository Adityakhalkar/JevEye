"use client";

import { RawImage } from "@huggingface/transformers";
import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";

import { Boxes, Toggle } from "@/components/Boxes";
import { CHANGE_THRESHOLD, LiveWindow, attendingFrom, toSample, type LiveFacts } from "@/lib/live";
import type { Found } from "@/lib/types";
import { Tracker, heading, type Track } from "@/lib/track";
import {
  DETECT_EDGE_LIVE,
  classify,
  detectObjects,
  embedImages,
  loadDetector,
  similarity,
  warmUp,
  type Choice,
  type LoadProgress,
} from "@/lib/vision";
import { VOCABULARIES, type VocabularyId } from "@/lib/vocab";

/** Sampling rate. Embedding costs ~15ms, so this leaves the tab responsive. */
const SAMPLE_MS = 300;
const FRAME_SIZE = 336;
/** About a minute of trace at the sampling rate. */
const TRACE_POINTS = 190;
/**
 * The detector runs every Nth sample, not every one.
 *
 * Embedding a frame costs ~15ms and detection costs closer to a second, so they
 * cannot share a cadence. Drift is measured on every frame because the gate
 * depends on it; naming is slower and reuses its last answer in between, which
 * is fine because what is in shot changes far more slowly than the view does.
 */
// Every second sample. Slower than that and association starts failing on
// anything quick, which costs identity to save work the settled-check already
// saves on a scene that is not moving.
const DETECT_EVERY = 2;
/** Boxes are redrawn this often, carried along their own velocity in between. */
const PREDICT_MS = 100;
/**
 * Below this much change, the detector is not run at all.
 *
 * It is the most expensive thing in the loop by an order of magnitude, and on a
 * scene that has not moved it returns what it returned last time. The gate that
 * decides when to ask Jev works just as well for deciding when to look.
 */
const REDETECT_CHANGE = 0.02;
/** At most this many things are followed; the rest is scenery. */
const ATTEND_TO = 3;

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
/** One plotted sample: how far the view had drifted, and whether that made Jev look. */
type Point = { at: number; change: number; asked: boolean };

const ATTENTION_LEVELS = ["nothing here", "worth noticing", "look now"];

export default function LivePage() {
  const video = useRef<HTMLVideoElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const window_ = useRef(new LiveWindow());
  const running = useRef(false);
  const inFlight = useRef(false);
  const ticks = useRef(0);
  const lastNaming = useRef<Choice | null>(null);
  const tracker = useRef(new Tracker());
  const lastDetectEmbed = useRef<Float32Array | null>(null);
  const [tracks, setTracks] = useState<Track[]>([]);
  const [scenery, setScenery] = useState(0);

  const [status, setStatus] = useState<"idle" | "loading" | "running">("idle");
  const [download, setDownload] = useState<LoadProgress | null>(null);
  const [facts, setFacts] = useState<LiveFacts | null>(null);
  const [trace, setTrace] = useState<Point[]>([]);
  /** The current frame's own label. The panel below reports the window instead. */
  const [now, setNow] = useState<{ label: string; p: number; named: boolean } | null>(null);
  /** The detector's most recent boxes, in the sampled frame's pixels. */
  const [found, setFound] = useState<{ boxes: Found[]; size: { width: number; height: number } } | null>(null);
  const [showBoxes, setShowBoxes] = useState(true);
  const [showNumbers, setShowNumbers] = useState(false);
  /** Which catalogue Jev says the watcher means; objects until told otherwise. */
  const [vocabulary, setVocabulary] = useState<VocabularyId>("objects");
  const vocabularyRef = useRef<VocabularyId>("objects");
  const [entries, setEntries] = useState<Entry[]>([]);
  const [watching, setWatching] = useState("");
  const [spend, setSpend] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const objectUrl = useRef<string | null>(null);
  const filePicker = useRef<HTMLInputElement>(null);
  const [sourceName, setSourceName] = useState<string | null>(null);

  const watchingRef = useRef(watching);
  useEffect(() => {
    watchingRef.current = watching;
  }, [watching]);

  useEffect(() => {
    vocabularyRef.current = vocabulary;
  }, [vocabulary]);

  /**
   * Ask Jev what the watcher means, once the typing stops.
   *
   * Debounced rather than per keystroke: this is a paid call, and "dog" on the
   * way to "dogwood flowers" is a different catalogue entirely.
   */
  useEffect(() => {
    const text = watching.trim();
    const timer = setTimeout(async () => {
      if (text === "") {
        setVocabulary("objects");
        return;
      }
      try {
        const response = await fetch("/api/vocabulary", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ watching: text }),
        });
        const json = await response.json();
        if (response.ok && json.vocabulary) setVocabulary(json.vocabulary);
      } catch {
        // Keeping the current catalogue is the right failure: the feed carries on.
      }
    }, 900);
    return () => clearTimeout(timer);
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
      if (!response.ok) throw new Error(json.error ?? "Jev could not be reached.");
      setEntries((prior) => [{ ...(json as Judgment), reason }, ...prior].slice(0, 20));
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

    const ratio = element.videoHeight / element.videoWidth || 0.75;
    surface.width = FRAME_SIZE;
    surface.height = Math.round(FRAME_SIZE * ratio);
    const context = surface.getContext("2d");
    if (!context) return;
    context.drawImage(element, 0, 0, surface.width, surface.height);

    /**
     * The whole frame only.
     *
     * Sampling the quarters as well and keeping the most confident of the five
     * cut unnameable samples from 17 in 30 to 1 in 19 — and made the naming
     * worse, not better: an agility course came back "sports ball" and
     * "frisbee" while the dog and handler went unmentioned. Picking the best of
     * five inflates confidence by selection alone, so the result was fabricated
     * certainty in place of an honest abstention. The wide-scene problem is
     * real, but it belongs to the vocabulary, not the sampling.
     */
    const frame = RawImage.fromCanvas(surface);
    const [embed] = await embedImages([frame]);

    /**
     * The detector names the frame; CLIP only names it when the detector found
     * nothing. On real footage a dog and handler in a wide shot came back
     * "frisbee" and "chair" from whole-frame classification, because the frame
     * genuinely is mostly grass. Localising the subject is the question worth
     * asking.
     */
    // Skip the detector outright when the view has not moved since it last ran.
    const settled =
      lastDetectEmbed.current !== null &&
      1 - similarity(embed, lastDetectEmbed.current) < REDETECT_CHANGE;

    let named = lastNaming.current;
    if ((ticks.current % DETECT_EVERY === 0 && !settled) || !named) {
      // Stricter than the still path: a moving wide shot throws up a lot of
      // true but irrelevant background, and the overlay has to stay readable.
      // Lower than it was: a small dog in a wide shot does not clear 0.7, and
      // the clutter a looser threshold admits is what salience is for.
      const detections = await detectObjects(frame, 0.5, DETECT_EDGE_LIVE);
      lastDetectEmbed.current = embed;
      tracker.current.setFrameArea(surface.width * surface.height);
      tracker.current.update(
        detections.map(({ label, score, box }) => ({ label, score, box })),
        Date.now(),
      );
      setFound({ boxes: [], size: { width: surface.width, height: surface.height } });
      if (detections.length > 0) {
        const ranked = detections
          .slice()
          .sort((a, b) => b.score - a.score)
          .map((d) => ({ label: d.label, p: d.score }));
        named = {
          label: ranked[0].label,
          confidence: ranked[0].p,
          unknown: false,
          source: "zero-shot",
          quality: null,
          probabilities: ranked,
        };
      } else {
        [named] = await classify([embed], VOCABULARIES[vocabularyRef.current]);
      }
      lastNaming.current = named;
    }
    ticks.current += 1;

    const now = Date.now();
    window_.current.push(toSample(now, embed, named));
    const attending = tracker.current.salient(now, ATTEND_TO);
    const setAside = tracker.current.scenery(now).length;
    const live = window_.current.facts(
      now,
      similarity,
      attending.map((t) => attendingFrom(t, now, heading(t))),
      setAside,
    );
    setScenery(setAside);
    const { judge: should, reason } = window_.current.shouldJudge(now, similarity);

    const best = named.probabilities[0];
    setNow(best ? { label: best.label, p: best.p, named: !named.unknown } : null);
    setFacts(live);
    setTrace((prior) =>
      [...prior, { at: now, change: live.changeSinceLastJudgment, asked: should }].slice(
        -TRACE_POINTS,
      ),
    );

    if (should) {
      window_.current.markJudged(now);
      void judge(live, reason);
    }
  }, [judge]);

  /**
   * Boxes move between detection passes.
   *
   * The detector speaks a couple of times a second; the eye notices far finer
   * steps than that. Carrying each track along its own velocity in between is
   * what turns a sequence of jumps into something that follows the subject.
   */
  useEffect(() => {
    if (status !== "running") return;
    const id = setInterval(
      () => setTracks(tracker.current.salient(Date.now(), ATTEND_TO)),
      PREDICT_MS,
    );
    return () => clearInterval(id);
  }, [status]);

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
        await new Promise((r) => setTimeout(r, Math.max(0, SAMPLE_MS - (Date.now() - started))));
      }
    };
    void loop();
    return () => {
      cancelled = true;
    };
  }, [status, tick]);

  /** Point the loop at a video file the viewer chose. It is read locally. */
  function openVideoFile(file: File | null | undefined) {
    if (!file) return;
    if (!file.type.startsWith("video/")) {
      setError("That is not a video this browser can play. MP4 or WebM work.");
      return;
    }
    if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
    objectUrl.current = URL.createObjectURL(file);
    setSourceName(file.name);
    void start(objectUrl.current);
  }

  async function start(source: "camera" | string) {
    setError(null);
    setStatus("loading");
    // A new source starts a new history; keeping the old trace would imply
    // drift between two unrelated scenes.
    window_.current = new LiveWindow();
    ticks.current = 0;
    lastNaming.current = null;
    tracker.current.reset();
    lastDetectEmbed.current = null;
    setTracks([]);
    setScenery(0);
    setFound(null);
    setTrace([]);
    setEntries([]);
    setFacts(null);
    setNow(null);
    await warmUp(setDownload);
    await loadDetector(setDownload);
    setDownload(null);

    const element = video.current;
    if (!element) return;
    try {
      if (source === "camera") {
        setSourceName("your camera");
        element.srcObject = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480 },
          audio: false,
        });
      } else {
        element.srcObject = null;
        element.src = source;
        element.loop = true;
      }
      await element.play();
    } catch (thrown) {
      setError(
        thrown instanceof Error
          ? `The video would not start. ${thrown.message}`
          : String(thrown),
      );
      setStatus("idle");
      return;
    }
    running.current = true;
    setStatus("running");
  }

  useEffect(
    () => () => {
      if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
    },
    [],
  );

  function stop() {
    running.current = false;
    (video.current?.srcObject as MediaStream | null)?.getTracks().forEach((t) => t.stop());
    if (video.current) video.current.srcObject = null;
    setStatus("idle");
  }

  const latest = entries[0];

  return (
    <main className="mx-auto min-h-screen max-w-5xl px-5 py-10 font-mono text-[13px] text-ink">
      <header className="mb-8 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-[22px] font-semibold tracking-tight text-ink">JevEye Live</h1>
          <p className="mt-1 max-w-[62ch] text-ink-2">
            It watches every moment, but only asks for a verdict when something actually changes.
            Nothing you point it at leaves this device.
          </p>
        </div>
        <label className="flex items-center gap-2 text-ink-3">
          Watching for
          <input
            value={watching}
            onChange={(e) => setWatching(e.target.value)}
            placeholder="anything worth noticing"
            className="w-56 rounded border border-rule bg-panel px-3 py-1.5 text-ink outline-none placeholder:text-ink-3 focus:border-trace"
          />
        </label>
      </header>

      <div className="grid items-start gap-4 lg:grid-cols-[1.6fr_1fr]">
        <figure
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            openVideoFile(e.dataTransfer.files?.[0]);
          }}
          className="relative overflow-hidden rounded-lg border border-rule bg-panel"
        >
          <span className="relative block">
            <video ref={video} playsInline muted className="block w-full" />
            {showBoxes && found && tracks.length > 0 && status === "running" && (
              <Boxes
                found={tracks.map((t) => ({
                  label: `${t.label} #${t.id}`,
                  score: t.score,
                  area: t.area,
                  box: t.box,
                }))}
                size={found.size}
              />
            )}
          </span>
          <canvas ref={canvas} className="hidden" />
          {now && status === "running" && (
            <figcaption className="absolute bottom-0 left-0 flex items-baseline gap-2 bg-ground/80 px-3 py-1.5 text-ink-2 backdrop-blur-sm">
              <span className={now.named ? "text-ink" : "text-ink-3"}>
                {now.named ? now.label : "nothing it can name"}
              </span>
              {now.named && (
                <span className="tabular-nums text-ink-3">{now.p.toFixed(2)}</span>
              )}
              <span className="text-ink-3">this frame</span>
            </figcaption>
          )}
          {status !== "running" && (
            <div className="flex aspect-video items-center justify-center px-6 text-center text-ink-3">
              {download
                ? `Loading the model, ${download.percent}% — first visit only`
                : status === "loading"
                  ? "Starting up"
                  : "Drop a video here, use your camera, or play the sample clip."}
            </div>
          )}
        </figure>

        <section className="rounded-lg border border-rule bg-panel p-4">
          <h2 className="mb-3 flex items-baseline justify-between text-ink-2">
            Following
            <span className="text-ink-3">
              {scenery > 0 ? `${scenery} more set aside as scenery` : `from ${vocabulary}`}
            </span>
          </h2>
          {tracks.length === 0 && (
            <p className="text-ink-3">Nothing moving yet — scenery is ignored.</p>
          )}
          <ul className="space-y-3">
            {tracks.slice(0, 5).map((t) => (
              <li key={t.id}>
                <div className="flex items-baseline justify-between gap-3">
                  <span className="truncate text-ink">
                    {t.label} <span className="text-ink-3">#{t.id}</span>
                  </span>
                  <span className="shrink-0 tabular-nums text-ink-3">
                    {((Date.now() - t.firstSeen) / 1000).toFixed(0)}s
                  </span>
                </div>
                <p className="text-ink-3">{heading(t)}</p>
              </li>
            ))}
          </ul>
          {facts && facts.unnamedSamples > 0 && (
            <p className="mt-3 text-ink-3">
              Nothing nameable in {facts.unnamedSamples} of {facts.samples} samples.
            </p>
          )}
        </section>
      </div>

      <Trace points={trace} />

      <section className="mt-4 grid items-start gap-4 lg:grid-cols-[1.6fr_1fr]">
        <div className="min-w-0 rounded-lg border border-rule bg-panel p-5">
          {latest ? (
            <>
              <p className="text-[clamp(20px,3.4vw,30px)] font-semibold leading-tight tracking-[-0.02em] text-ink">
                {latest.situation}
              </p>
              <p className="mt-2 text-ink-2">
                Asked because {latest.reason}.
              </p>
              <Attention level={latest.attention} />
            </>
          ) : (
            <p className="text-ink-3">
              {status === "running"
                ? "Watching. The first verdict takes a few seconds."
                : "Start the video and the first verdict appears here."}
            </p>
          )}
        </div>

        <div className="min-w-0 rounded-lg border border-rule bg-panel p-4">
          <h2 className="mb-3 flex items-baseline justify-between text-ink-2">
            Earlier
            <span className="tabular-nums text-ink-3">
              {showNumbers ? `$${spend.toFixed(6)} so far` : ""}
            </span>
          </h2>
          {entries.length <= 1 && <p className="text-ink-3">Nothing yet.</p>}
          <ol className="space-y-1.5">
            {entries.slice(1, 7).map((e) => (
              <li key={e.at} className="flex min-w-0 gap-3">
                <span className="shrink-0 tabular-nums text-ink-3">
                  {new Date(e.at).toLocaleTimeString([], { hour12: false })}
                </span>
                <span className="truncate text-ink-2">{e.situation}</span>
              </li>
            ))}
          </ol>
        </div>
      </section>

      <div className="mt-6 flex flex-wrap items-center gap-2">
        {status === "running" ? (
          <button
            onClick={stop}
            className="rounded bg-ink px-4 py-2 font-semibold text-ground focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-trace"
          >
            Stop watching
          </button>
        ) : (
          <>
            <button
              onClick={() => start("camera")}
              className="rounded bg-ink px-4 py-2 font-semibold text-ground focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-trace"
            >
              Use my camera
            </button>
            <button
              onClick={() => filePicker.current?.click()}
              className="rounded border border-rule px-4 py-2 text-ink-2 hover:border-ink-3 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-trace"
            >
              Open a video file
            </button>
            <button
              onClick={() => start("/sample.mp4")}
              className="rounded border border-rule px-4 py-2 text-ink-2 hover:border-ink-3 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-trace"
            >
              Play the sample clip
            </button>
          </>
        )}
        <input
          ref={filePicker}
          type="file"
          accept="video/*"
          className="hidden"
          onChange={(e) => openVideoFile(e.target.files?.[0])}
        />
        {sourceName && <span className="text-ink-3">watching {sourceName}</span>}
      </div>

      {error && (
        <p role="alert" className="mt-4 rounded border border-alarm/40 bg-alarm/10 px-3 py-2 text-alarm">
          {error}
        </p>
      )}

      <div className="mt-6 flex flex-wrap gap-5">
        <Toggle checked={showBoxes} onChange={setShowBoxes}>
          Outline what it finds
        </Toggle>
        <Toggle checked={showNumbers} onChange={setShowNumbers}>
          Show the numbers behind this
        </Toggle>
      </div>

      <footer className="mt-6 max-w-[70ch] text-ink-3">
        Video is read on this device and never uploaded. Only a short written summary of what
        changed is sent away to be judged.{" "}
        <Link href="/" className="text-ink-2 underline underline-offset-2 hover:text-ink">
          Ask about a still image instead
        </Link>
      </footer>
    </main>
  );
}

/**
 * The gate, drawn.
 *
 * Drift climbs while the view changes and drops to nothing each time Jev looks,
 * so the sawtooth is the mechanism rather than an illustration of it.
 */
function Trace({ points }: { points: Point[] }) {
  const box = useRef<HTMLDivElement>(null);
  const [W, setW] = useState(960);
  useEffect(() => {
    const element = box.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => setW(Math.round(entry.contentRect.width)));
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  const H = 120;
  const PAD = 10;
  const ceiling = Math.max(CHANGE_THRESHOLD * 2, ...points.map((p) => p.change), 0.2) * 1.1;
  const x = (i: number) => (i / Math.max(TRACE_POINTS - 1, 1)) * W;
  const y = (v: number) => H - PAD - (Math.min(v, ceiling) / ceiling) * (H - PAD * 2);

  const path = points.map((p, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${y(p.change).toFixed(1)}`).join(" ");
  const asked = points.map((p, i) => ({ ...p, i })).filter((p) => p.asked);
  const current = points.at(-1)?.change ?? 0;

  return (
    <section ref={box} className="mt-4 min-w-0 rounded-lg border border-rule bg-panel px-4 pb-3 pt-4">
      <h2 className="mb-1 flex flex-wrap items-baseline justify-between gap-2 text-ink-2">
        How much the view has changed since the last verdict
        <span className="tabular-nums text-ink-3">
          a dot marks each time it asked
        </span>
      </h2>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        width={W}
        height={H}
        className="w-full"
        role="img"
        aria-label={`Drift is ${current.toFixed(3)}. Jev is asked when it passes ${CHANGE_THRESHOLD}. ${asked.length} readings in this stretch.`}
      >
        <line
          x1="0"
          x2={W}
          y1={y(CHANGE_THRESHOLD)}
          y2={y(CHANGE_THRESHOLD)}
          stroke="var(--color-watch)"
          strokeWidth="2"
          strokeDasharray="6 6"
        />
        {points.length > 1 && (
          <path d={path} fill="none" stroke="var(--color-trace)" strokeWidth="2" strokeLinejoin="round" />
        )}
        {asked.map((p) => (
          <g key={p.at}>
            <line x1={x(p.i)} x2={x(p.i)} y1={PAD} y2={H - PAD} stroke="var(--color-watch)" strokeWidth="1" opacity="0.45" />
            <circle cx={x(p.i)} cy={y(p.change)} r="4.5" fill="var(--color-watch)" stroke="var(--color-panel)" strokeWidth="2">
              <title>{`Asked Jev at drift ${p.change.toFixed(3)}`}</title>
            </circle>
          </g>
        ))}
      </svg>
      <p className="text-ink-3">
        {points.length < 2
          ? "The line starts once the video does."
          : `The line climbs as the scene changes and falls back to nothing each time it asks — ${asked.length} verdicts so far.`}
      </p>
    </section>
  );
}

/** An ordered rubric, so segments plus the level's own words — never colour alone. */
function Attention({ level }: { level: number }) {
  const step = Math.min(Math.max(Math.round(level), 0), 2);
  const tone = step === 2 ? "bg-alarm" : step === 1 ? "bg-watch" : "bg-ink-3";
  return (
    <p className="mt-4 flex items-center gap-3 text-ink-2">
      <span className="flex gap-1" aria-hidden>
        {[0, 1, 2].map((i) => (
          <span key={i} className={`h-2.5 w-7 rounded-sm ${i <= step ? tone : "bg-rule"}`} />
        ))}
      </span>
      {ATTENTION_LEVELS[step]}
      <span className="tabular-nums text-ink-3">{level.toFixed(2)}</span>
    </p>
  );
}
