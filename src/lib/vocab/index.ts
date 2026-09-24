import { COCO_80 } from "./coco80.ts";
import { FLOWERS_102 } from "./flowers102.ts";

/** A shipped label set. Labels are never generated: Jev picks the set, CLIP scores within it. */
export type Vocabulary = {
  id: VocabularyId;
  /** Shown to Jev when it decides which set a question is about. */
  description: string;
  /** The noun used as the open-vocabulary detection prompt. */
  instanceNoun: string;
  labels: readonly string[];
  /**
   * CLIP prompt templates; the label replaces `{}`.
   *
   * Several rather than one: CLIP's own paper ensembles eighty and reports
   * several points of accuracy for it, because any single phrasing is a quirk
   * of wording as much as a description. The embeddings are averaged once and
   * cached, so a longer list costs nothing after warm-up.
   */
  hypotheses: readonly string[];
  /**
   * Base path of a fitted linear probe, served from `public/`. When present the
   * classifier uses it instead of zero-shot text scoring; when the files are
   * missing the zero-shot path still works, so this is safe to point at nothing.
   */
  probe?: string;
};

export type VocabularyId = "flowers" | "objects";

export const VOCABULARIES: Record<VocabularyId, Vocabulary> = {
  flowers: {
    id: "flowers",
    description:
      "Species of flower or flowering plant — 102 garden and wild species, e.g. corn poppy, oxeye daisy, sunflower, rose.",
    instanceNoun: "flower",
    labels: FLOWERS_102,
    hypotheses: [
      "a photo of a {}, a type of flower.",
      "a close-up photo of a {}.",
      "a photo of the flower {}.",
      "a bright photo of a {}, a type of flower.",
      "a cropped photo of a {}.",
      "a photo of many {} together.",
    ],
    probe: "/probes/flowers",
  },
  objects: {
    id: "objects",
    description:
      "Everyday objects, animals, vehicles, food and people — 80 common categories, e.g. person, dog, car, pizza, laptop.",
    instanceNoun: "object",
    labels: COCO_80,
    hypotheses: [
      "a photo of a {}.",
      "a photo of the {}.",
      "a close-up photo of a {}.",
      "a bright photo of a {}.",
      "a cropped photo of a {}.",
      "a photo of a {} in the real world.",
    ],
  },
};

export { COCO_80, FLOWERS_102 };
