/**
 * Ordered scales for `score`, shipped as data.
 *
 * Jev picks which scale a question is about, the same way it picks a
 * vocabulary. It cannot write one: nothing in this system generates text, so
 * every level below is a sentence CLIP will be asked to score, written here in
 * advance.
 *
 * Levels run worst-to-best, or least-to-most, and are phrased as descriptions
 * of a photograph because that is what CLIP was trained to match.
 */
export type ScaleId = "health" | "sharpness" | "crowding" | "damage" | "lighting";

export type Scale = {
  id: ScaleId;
  /** Shown to Jev when it decides which scale a question is about. */
  description: string;
  /** At least three, lowest first. */
  levels: readonly string[];
};

export const SCALES: Record<ScaleId, Scale> = {
  health: {
    id: "health",
    description:
      "How healthy or alive something looks — a plant, a crop, a leaf. Questions like 'does this look healthy', 'is this plant dying'.",
    levels: [
      "a photograph of a dead or dying plant, brown, withered and collapsed",
      "a photograph of a plant in fair condition, with some yellowing, wilting or damage",
      "a photograph of a healthy, vigorous plant in full colour",
    ],
  },
  sharpness: {
    id: "sharpness",
    description:
      "How sharp or blurry the photograph itself is. Questions like 'is this blurry', 'is this in focus', 'is this a good photo'.",
    levels: [
      "a very blurry, out-of-focus, smeared photograph",
      "a slightly soft photograph, not perfectly sharp",
      "a crisp, sharply focused photograph with fine detail",
    ],
  },
  crowding: {
    id: "crowding",
    description:
      "How full or busy the scene is. Questions like 'is it crowded', 'is this busy', 'how packed is it'.",
    levels: [
      "a photograph of an empty, bare scene with almost nothing in it",
      "a photograph of a moderately occupied scene with some things in it",
      "a photograph of a dense, crowded scene packed with things",
    ],
  },
  damage: {
    id: "damage",
    description:
      "How damaged or worn something is. Questions like 'is this broken', 'does this look damaged', 'what condition is it in'.",
    levels: [
      "a photograph of something pristine and undamaged, in new condition",
      "a photograph of something with light wear, small marks or scratches",
      "a photograph of something badly damaged, broken or falling apart",
    ],
  },
  lighting: {
    id: "lighting",
    description:
      "How bright or well-lit the photograph is. Questions like 'is this too dark', 'was this taken at night', 'is it well lit'.",
    levels: [
      "a very dark photograph, underexposed, taken at night or in deep shadow",
      "a dimly lit photograph, somewhat underexposed",
      "a bright, evenly lit photograph taken in good light",
    ],
  },
};
