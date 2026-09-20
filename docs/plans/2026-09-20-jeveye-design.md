# JevEye design

2026-09-20

## Thesis

Jev accepts text only. That is a design decision, not a gap: it is a System One model over state, not a vision-language model. So a vision layer for Jev cannot be a wrapper that smuggles pixels in. It has to turn pixels into state honestly, with calibrated numbers going in so that Jev's calibrated numbers coming out still mean something.

Two layers, strict boundary:

- **Vision** answers questions of fact — is a thing present, which of these labels, how far along this scale, how much of the picture it covers. Every answer carries a calibrated probability or abstains. It never judges.
- **Judgment** is Jev. It reads the fact sheet as text with every confidence attached, and never sees pixels.

The loop closes because Jev also chooses the probes: given the question it picks the label set and the reading before any pixel is touched. This is the same two-role structure JevNQL uses — Jev decides what a question means, then judges meaning in the data.

A VLM collapses seeing and judging into one opaque pass, and its confidence concerns tokens rather than the world. Here both halves report their own uncertainty and the fact sheet between them is readable.

## Primitives

Three mirror Jev's own names and return types, so half the API is already familiar:

```
detect  (image, statement)      -> Probability | Unknown     (Jev's noul)
choose  (image, labels)         -> Label + Distribution      (Jev's choice)
score   (image, levels)         -> Position(0..1) + Conf     (Jev's score)
coverage(image, thing)          -> Distribution over tiles
judge   (image, question)       -> Answer + FactSheet        (vision, then Jev)
```

Two commitments:

**Counting returns a distribution, not an integer.** A point estimate has to pick a presence threshold and then lie about the marginal cases either way. The Poisson-binomial over per-observation probabilities widens automatically when the evidence is marginal.

**Every primitive may abstain.** Below its reliability bar a primitive returns unknown rather than a shrug-shaped 0.5, and `unknown` reaches Jev as the literal word, so Jev judges a missing fact instead of a fabricated coin-flip. A VLM cannot express this; it always produces tokens.

`judge` is the only primitive that costs money. Vision runs locally and free; a fact sheet is a few hundred tokens. Same cost shape as JevNQL: the expensive layer sees a small, distilled input.

## Abstention

A flat confidence floor cannot serve both an 80-way and a 102-way choice: the same 0.3 is lax for two labels and nearly unreachable for a hundred, where even a correct top-1 is diffuse. So the rule is relative:

```
reportable  ⟺  top ≥ min(12 × 1/N, 0.5)  ∧  top ≥ 1.5 × runnerUp
```

Comfortably above chance, and clear of the runner-up — which is what "this label, not that one" means. The ceiling exists so a two-way choice can still pass.

Tallies are weighted by confidence, not by tile count, so a label seen faintly in four tiles does not outrank one seen clearly in three.

## Shape

Next.js, one Vercel project. Vision in the browser via transformers.js; Jev in a route handler because the API key cannot ship to a client.

```
browser ── CLIP q8, WebGPU (WASM fallback)
           detect / choose / score / coverage
              │  fact sheet, a few hundred bytes
              ▼
POST /api/plan, /api/judge ── @typesafe-ai/sdk, key server-side only
```

Consequence worth keeping: the image never leaves the browser. Only the fact sheet crosses the network.

## Failure modes

| Case | Behavior |
|---|---|
| Question is not about the image | A noul gates the pipeline. Below 0.5: say so, run no probes, spend nothing on judging. |
| Nothing found | Fact sheet reports it; Jev is told plainly; the answer says it cannot see the subject. |
| Subject outside the label set | The real hazard — CLIP reaches for the nearest of its labels. The chance-and-margin rule turns that into an abstention rather than a confident lie. |
| Jev cannot settle it | An explicit "cannot tell from these observations" option. Without it, a single surviving candidate makes the choice trivial and Jev reports 1.00 for one shaky tile. |

Plus: no WebGPU → WASM; download failure → retryable; images downscaled to 1280 px; HEIC unsupported in some browsers, said out loud; Jev 429 → SDK backoff.

## What changed during implementation

The design above is what survived contact. Three things did not:

1. **CLIP is ViT-B/32, not RN50.** The open-vocabulary requirement — scoring arbitrary text against an image — is what picks the model, and ONNX availability in the browser picks the variant. "CNN" lost to what actually loads.
2. **`locate` and `count` became `coverage`.** OWL-ViT's `class_head` Cast node has no ORT-web implementation at q8, q4f16 or fp16; only its 583 MB fp32 graph could load. Instance counting is gone; the picture is cut into a 4×4 grid instead, and "14 of 16 tiles" measures how much of the image a kind covers. Arguably better on dense scenes where boxes overlap anyway, but it is a different claim and is labelled as one.
3. **Calibration ships unfitted.** The plumbing is there — temperatures, reliability bars, ECE-ready reliability tables — but no offline fit has been run, so temperatures are identity and the bars are hand-set. The 12× chance multiple was chosen by eye against a single photograph. The success criterion of ECE ≤ 0.05 per primitive is therefore **not met**, and the UI says so on every answer.

`compare` (two-image questions) was designed and not built.

## Verified

End to end in a real browser against live Jev, on a mixed field of poppies and camomile:

```
"what type of flower is it?"
  jev reads:  on topic 0.77 · flowers (1.00) · reading "only" (0.96) · $0.000024
  vision:     4.9 s, 14 of 16 tiles hold flowers
              corn poppy weight 0.96 · ball moss 0.20 · mexican aster 0.14
              8 tiles could not be named
  jev judges: corn poppy · confidence 0.37 · support 0.88 · mixed 0.71 · $0.000035

"what is the capital of France?"
  on topic 0.04 → refused, no probes, nothing spent
```

Jev reads "what type of flower **is it**" as `only` rather than `dominant` — better English than the design assumed. The mixed field then surfaces through the `mixed` noul instead of through the reading.

## Follow-up: the reading now drives the probes

The first cut had a gap between what this document claimed and what ran. `probe()`
ignored the reading entirely — the same three probes fired whatever Jev decided the
question meant, and the reading only coloured the judging. `score` was implemented,
exported, and never called.

Closed by making the reading select the probe set, which also makes the two
open-vocabulary primitives reachable:

| Reading | Probes | Cost |
|---|---|---|
| `only` / `dominant` / `every` | coverage + classifier over tiles | ~4.9 s |
| `presence` | coverage + one `detect` on a Jev-chosen subject | ~3.0 s |
| `count` | coverage only | ~2.7 s |
| `rating` | one `score` over a shipped scale | ~2.6 s |

Two consequences worth recording. Readings that skip the 102-way classifier are
roughly twice as fast. And `presence` and `rating` never touch a label set, so they
work on subjects no vocabulary covers — the closed set constrains naming, not
looking.

Scales had to become shipped data for the same reason vocabularies did: Jev picks
one, it cannot write one. Five ship: health, sharpness, crowding, damage, lighting.
`presence` needs a second Jev call, because the labels to choose a subject from are
only known once the first call has settled the vocabulary.

Verified in-browser on the same photograph: `rating`/health landed at 0.77 with Jev
answering "healthy, vigorous plant" at 0.63; `presence`/dog returned "no — no dog" at
0.66 with the detector abstaining and 0 of 5 checked tiles matching; `count` gave 14
of 16 tiles at dependability 0.34.

One bug the work surfaced and fixed: presence reported matching tiles out of 16 when
only the 5 occupied tiles were ever checked. A denominator that flatters the evidence
is exactly the failure this project exists to avoid.

## Follow-up: a trained probe, and calibration that is finally measured

The design shipped with calibration plumbing and no fit, which made the central
claim architectural rather than demonstrated. That is now closed for one path.

A linear probe — logistic regression on CLIP's frozen 512-d embeddings, fitted on
Flowers-102 train+val (2,040 images) — replaces zero-shot text scoring when a
vocabulary has one. A temperature is fitted separately on 1,200 held-out images,
and both are reported on a further 1,500 that neither ever saw.

| | zero-shot | probe |
|---|---|---|
| accuracy | 29.1% | 92.8% |
| ECE raw | 0.093 | 0.459 |
| ECE calibrated | 0.041 | 0.014 |

The ECE ≤ 0.05 criterion in §6 is met for flower naming. It remains unmet for
everything else, and the README says so rather than generalizing one result.

Four things learned:

**Zero-shot here is 29%, not the ~66% the CLIP paper reports.** q8 quantization
and a single prompt template instead of an 80-template ensemble account for the
gap. Quoting a paper's number for your own build is how you end up with a
baseline nobody can reproduce.

**The raw probe was overconfident in reverse** — ECE 0.459 while 92.8% accurate,
far too timid. A temperature below 1 sharpens it. Calibration is a two-way
correction, not a way of making models humbler.

**The baseline doubled as a correctness check.** The 102 label names were written
from memory; had their order not matched the dataset's class indices, zero-shot
accuracy would have sat at chance (1.0%) instead of 29%. A baseline you can
reason about catches bugs a test would not have been written for.

**Serving skew was avoided by construction.** `tools/embed.mjs` loads the same
checkpoint at the same quantization the browser does, so the probe is fitted on
the embeddings it will actually meet. Fitting on fp32 and serving q8 is a silent
distribution shift.

This also forced a refactor worth keeping: CLIP's towers now load separately
instead of behind a pipeline, so an image's embedding is available directly. The
probe and the zero-shot classifier share one forward pass, and text embeddings
are cached rather than recomputed per call.

On the demo photograph the improvement is legible rather than merely numeric: the
white flowers are now named *oxeye daisy* — the nearest species in the vocabulary
to the scentless mayweed actually present — so Jev's mixed-picture probability
rises from 0.71 to 0.89. The second species finally reaches the evidence.

## Not built

- Fitted calibration for anything but flower naming: `detect`, `score`, `coverage`
  and the COCO vocabulary still ship identity temperatures.
- `compare`, and any two-image question.
- Overlapping or multi-scale tiles.
- Golden-file fact-sheet tests and a Playwright end-to-end test. The calibration arithmetic is unit-tested; the model path was verified by hand.
