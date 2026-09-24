/**
 * Does Jev read questions the way the pipeline needs them read?
 *
 * The planner decides which probes run, so a misread question is a wrong answer
 * no amount of vision quality can rescue. This is the one component whose
 * behaviour changes without the code changing, which is exactly why it needs a
 * fixed set of cases rather than a memory of it working once.
 *
 * Where more than one reading is defensible the case accepts a set: "what kind
 * of flower is this" is legitimately `only` or `dominant`, and scoring it as a
 * single right answer would measure the rubric rather than the model.
 */
const BASE = process.env.EVAL_BASE ?? "http://localhost:3117";

/** reading: accepted set. subject/scale/vocabulary: expected, or null to skip. */
const CASES = [
  // naming
  { q: "what type of flower is it?", reading: ["only", "dominant"], vocabulary: "flowers" },
  { q: "what kind of flower am I looking at?", reading: ["only", "dominant"], vocabulary: "flowers" },
  { q: "which flowers are in this field?", reading: ["every", "dominant"], vocabulary: "flowers" },
  { q: "what is this?", reading: ["only", "dominant"] },
  { q: "what is in this picture?", reading: ["only", "dominant", "every"] },

  // presence
  { q: "is there a dog in this photo?", reading: ["presence"], subject: "dog", vocabulary: "objects" },
  { q: "is there a cat here?", reading: ["presence"], subject: "cat", vocabulary: "objects" },
  { q: "can you see a bicycle?", reading: ["presence"], subject: "bicycle", vocabulary: "objects" },
  { q: "is anyone in the room?", reading: ["presence"], subject: "person", vocabulary: "objects" },
  { q: "is there a laptop on the desk?", reading: ["presence"], subject: "laptop", vocabulary: "objects" },

  // counting
  { q: "how many dogs are there?", reading: ["count"], subject: "dog", vocabulary: "objects" },
  { q: "how many people are in this picture?", reading: ["count"], subject: "person", vocabulary: "objects" },
  { q: "count the cars", reading: ["count"], subject: "car", vocabulary: "objects" },
  { q: "how many chairs can you see?", reading: ["count"], subject: "chair", vocabulary: "objects" },

  // rating
  { q: "do these plants look healthy?", reading: ["rating"], scale: "health" },
  { q: "is this photo blurry?", reading: ["rating"], scale: "sharpness" },
  { q: "is this image in focus?", reading: ["rating"], scale: "sharpness" },
  { q: "how crowded is this scene?", reading: ["rating"], scale: "crowding" },
  { q: "is this thing damaged?", reading: ["rating"], scale: "damage" },
  { q: "was this taken in the dark?", reading: ["rating"], scale: "lighting" },

  // not about the picture at all
  { q: "what is the capital of France?", offTopic: true },
  { q: "what is 17 times 4?", offTopic: true },
  { q: "who wrote Hamlet?", offTopic: true },
  { q: "what time does the shop close?", offTopic: true },
];

async function plan(question) {
  const response = await fetch(`${BASE}/api/plan`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!response.ok) throw new Error(`${response.status} for "${question}"`);
  return response.json();
}

export async function evaluatePlanner() {
  const score = {
    onTopic: { right: 0, of: 0 },
    reading: { right: 0, of: 0 },
    vocabulary: { right: 0, of: 0 },
    subject: { right: 0, of: 0 },
    scale: { right: 0, of: 0 },
  };
  const wrong = [];
  let tokens = 0;

  for (const c of CASES) {
    let p;
    try {
      p = await plan(c.q);
    } catch (e) {
      wrong.push(`${c.q} — request failed: ${e.message}`);
      continue;
    }
    tokens += p.inputTokens ?? 0;

    if (c.offTopic) {
      score.onTopic.of += 1;
      if (!p.proceed) score.onTopic.right += 1;
      else wrong.push(`"${c.q}" was treated as a question about the picture (${p.onTopic})`);
      continue;
    }

    score.onTopic.of += 1;
    if (p.proceed) score.onTopic.right += 1;
    else wrong.push(`"${c.q}" was refused as off-topic (${p.onTopic})`);

    score.reading.of += 1;
    if (c.reading.includes(p.reading)) score.reading.right += 1;
    else wrong.push(`"${c.q}" read as ${p.reading}, expected ${c.reading.join(" or ")}`);

    if (c.vocabulary) {
      score.vocabulary.of += 1;
      if (p.vocabulary === c.vocabulary) score.vocabulary.right += 1;
      else wrong.push(`"${c.q}" chose the ${p.vocabulary} catalogue, expected ${c.vocabulary}`);
    }
    if (c.subject) {
      score.subject.of += 1;
      if (p.subject === c.subject) score.subject.right += 1;
      else wrong.push(`"${c.q}" looked for ${p.subject}, expected ${c.subject}`);
    }
    if (c.scale) {
      score.scale.of += 1;
      if (p.scale === c.scale) score.scale.right += 1;
      else wrong.push(`"${c.q}" used the ${p.scale} scale, expected ${c.scale}`);
    }
  }

  /**
   * Below these, something has regressed.
   *
   * Every field reads 100% today, but the floors keep slack: Jev is a model,
   * not a lookup, and a suite that fails on one borderline answer gets muted
   * rather than fixed.
   */
  const FLOOR = { onTopic: 0.95, reading: 0.9, vocabulary: 0.95, subject: 0.9, scale: 0.85 };
  let failures = 0;

  console.log("\nHOW JEV READS QUESTIONS");
  for (const [field, s] of Object.entries(score)) {
    if (s.of === 0) continue;
    const rate = s.right / s.of;
    const ok = rate >= FLOOR[field];
    if (!ok) failures += 1;
    console.log(
      `  ${ok ? "pass" : "FAIL"}  ${field.padEnd(12)} ${s.right}/${s.of} ` +
        `(${(rate * 100).toFixed(0)}%, floor ${FLOOR[field] * 100}%)`,
    );
  }
  if (wrong.length > 0) {
    console.log("  misreadings:");
    for (const w of wrong) console.log(`    ${w}`);
  }
  console.log(`  cost: $${((tokens / 1e6) * 0.042).toFixed(6)} across ${CASES.length} questions`);
  return failures;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  process.exit((await evaluatePlanner()) > 0 ? 1 : 0);
}
