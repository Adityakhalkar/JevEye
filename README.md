# JevEye

**A CNN reports what it sees, with a calibrated confidence or an abstention. Jev judges what that means.**

<p align="center">
  <img src="docs/jeveye-flowers.png" alt="JevEye answering a question about a field of poppies and camomile: what Jev read into the question, the vision probes, the answer, and the fact sheet behind it" width="820">
  <br><sub>Photograph by JackyM59, <a href="https://creativecommons.org/licenses/by-sa/4.0">CC BY-SA 4.0</a>, via Wikimedia Commons. See <a href="fixtures/README.md">fixtures</a>.</sub>
</p>

[Jev](https://docs.typesafe.ai) accepts text only — *"State must be a string, JSON object, or array of text values. Images, audio, and video are not supported (yet)."* JevEye is the layer that lets you ask it about a photograph anyway, without pretending the model can see.

Two layers, strict boundary:

- **The vision layer** answers only questions of fact. Is this a flower; which of these 102 species; how much of the picture holds one. Every answer carries a probability, or abstains. It never judges.
- **Jev** judges. It reads the facts as text, with every confidence still attached — `corn poppy: 3 tiles, mean 0.32, weight 0.96` — and draws the conclusion. It never sees pixels.

The loop closes because Jev decides *what to look for*. Before a single pixel is touched it reads your question into one of six **readings**, and the reading chooses which probes run:

| You ask | Reading | What actually runs |
|---|---|---|
| "what type of flower is it?" | `only` | coverage, then the classifier over tiles |
| "which flowers are in this field?" | `every` | same, judged for multiplicity |
| "is there a dog in this photo?" | `presence` | Jev picks `dog` from the label set; one open-vocabulary sentence is scored |
| "how much is covered in flowers?" | `count` | coverage only — the classifier never runs |
| "do these plants look healthy?" | `rating` | an ordered scale; no label set involved at all |
| "is this photo blurry?" | `rating` | the `sharpness` scale |

Two of those paths never touch the 182 labels. `detect` and `score` take **arbitrary sentences**, so presence and rating are open-vocabulary — the closed set only constrains *naming*.

```text
drop field.jpg  +  "what type of flower is it?"

JEV READS THE QUESTION                            $0.000024
  on topic      0.77
  vocabulary    flowers (1.00) — 102 labels
  reading       only (0.96)

VISION PROBES                            4.9 s, in the browser
  coverage "flower" over a 4×4 grid
  choose ×14 over 102 flowers labels

WHAT IT SAW
  tiles holding the subject   14 of 16 (90% between 12 and 15)
  corn poppy                  3 tiles · mean 0.32 · weight 0.96
  ball moss                   1 tiles · mean 0.20 · weight 0.20
  mexican aster               1 tiles · mean 0.14 · weight 0.14
  common dandelion            1 tiles · mean 0.12 · weight 0.12
  could not be named          8 tiles

JEV JUDGES                                        $0.000035
  corn poppy    confidence 0.37    support 0.88    mixed 0.71

  "Found in most likely 14 of 16 tiles. More than one kind covers a
   meaningful part of the picture (0.71), so this is not a single-kind image."
```

Why this is not just a VLM with extra steps: a VLM collapses seeing and judging into one opaque pass, and its confidence is about tokens rather than about the world. Here the seeing and the judging are separate, each reports its own uncertainty, and the fact sheet between them is readable — you can see what it thought it saw before it concluded anything.

## Run it

Needs Node 20+ and a [TypeSafe](https://typesafe.ai) API key.

```bash
npm install
echo "TYPESAFE_API_KEY=your-key" > .env.local
npm run dev
```

Deploying: one Vercel project, `TYPESAFE_API_KEY` set as an environment variable. Nothing else to host — the models run in the visitor's browser.

```bash
vercel deploy
```

## How it works

| Layer | Where | What |
|---|---|---|
| CLIP ViT-B/32, q8 ONNX | the browser, via transformers.js on WebGPU (WASM fallback) | `detect`, `choose`, `score`, `coverage` |
| Jev, via `@typesafe-ai/sdk` | `/api/plan`, `/api/judge` | picks the reading, label set, subject and scale; judges the fact sheet |

Each reading produces a differently shaped fact sheet, and each gets its own question of Jev — naming is a `choice`, presence is a `noul`, rating is a `score`. The one constant is that Jev is always told how thin the evidence was, which is why a "no" can come back at 0.66 with support 0.30.

**The image never leaves the browser.** Only the fact sheet — a few hundred bytes of JSON — crosses the network. The API key is the reason a server exists at all.

**First visit downloads about 40 MB** of quantized CLIP weights, cached by the browser thereafter.

Nothing in the vision layer is new. CLIP is
[OpenAI's](https://arxiv.org/abs/2103.00020) (Radford et al., 2021), used as released, via
[Xenova's ONNX export](https://huggingface.co/Xenova/clip-vit-base-patch32) and
[transformers.js](https://github.com/huggingface/transformers.js). The
`"a photo of a {}"` prompt template is that paper's own zero-shot recipe.
[Temperature scaling and ECE](https://arxiv.org/abs/1706.04599) are Guo et al., 2017;
abstaining below a threshold is [Chow's reject option](https://doi.org/10.1109/TIT.1970.1054406), 1970.
What is JevEye's own is the arrangement: the strict split between scoring facts and judging them,
Jev choosing the probes before any pixel is read, the chance-relative abstention rule, and routing
tile presences through a Poisson-binomial so counts arrive as intervals.

Label sets **and rating scales** are shipped, never generated — Jev picks which one applies, exactly as it picks a vocabulary: [Oxford Flowers-102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) and the 80 COCO categories. Jev picks which one a question is about. Nothing in this system generates text.

## What it does not do

Stated plainly, because the whole point is honest uncertainty.

- **Calibration is unfitted.** `src/lib/calibration.ts` ships identity temperatures, so the confidences are raw model outputs and are very likely overconfident. `detect` in particular returns 1.00 far too readily. Fitting needs a labelled held-out split and an offline run that has not happened. Until then the abstention thresholds are hand-set, and the 12× chance bar for `choose` was picked by eye against one photograph — which is not validation.
- **It counts coverage, not instances.** The original design used OWL-ViT for `locate` and `count`. Its `class_head` Cast node has no ONNX Runtime Web implementation at q8, q4f16 or fp16, and only the 583 MB fp32 graph could load — too much for a browser. So the picture is cut into a 4×4 grid and each tile examined separately. "14 of 16 tiles" measures how much of the image a kind covers, not how many flowers there are.
- **The grid is coarse.** A flower straddling two tiles is seen twice; one much smaller than a tile is diluted by its background.
- **`compare` is not implemented.** Two-image questions ("is this the same plant?") are designed but not built.
- **Rating uses five shipped scales** — health, sharpness, crowding, damage, lighting. Ask along a scale that isn't there and it falls back to naming, because Jev cannot write a new scale.
- **Out-of-vocabulary is the real hazard.** Show it a protea and CLIP will reach for the nearest of its 102 labels. The margin-and-chance abstention rule is what keeps that from becoming a confident lie, and it is exactly the part that fitting would make trustworthy.
- **Domain drift.** The label sets and thresholds suit web-like photographs. Satellite, medical, document and screenshot images will be wrong in ways the confidences will not warn you about.

## Tests

```bash
npm test
```

Covers the calibration arithmetic: temperature scaling, the Poisson-binomial over tile presences, the interval that widens on marginal evidence, and the chance-relative abstention rule. The model path is not unit-tested; it was verified end to end in a browser.

## Layout

```text
src/lib/calibration.ts   temperatures, abstention rules, Poisson-binomial
src/lib/vision.ts        the primitives and the fact sheet, browser-side
src/lib/vocab/           shipped label sets
src/lib/scales.ts        shipped rating scales
src/app/api/plan/        Jev reads the question
src/app/api/judge/       Jev judges the fact sheet
src/app/page.tsx         drop zone, chat bar, answer, fact sheet
```

Sibling project to [JevNQL](https://github.com/Adityakhalkar/JevNQL), which does the same trick for tabular data: send the arithmetic to a relational engine, send only the judgments to Jev.

MIT.
