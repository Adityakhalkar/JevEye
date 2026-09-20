import { COCO_80 } from "./coco80";
import { FLOWERS_102 } from "./flowers102";

/** A shipped label set. Labels are never generated: Jev picks the set, CLIP scores within it. */
export type Vocabulary = {
  id: VocabularyId;
  /** Shown to Jev when it decides which set a question is about. */
  description: string;
  /** The noun used as the open-vocabulary detection prompt. */
  instanceNoun: string;
  labels: readonly string[];
  /** CLIP prompt template; the label replaces `{}`. */
  hypothesis: string;
};

export type VocabularyId = "flowers" | "objects";

export const VOCABULARIES: Record<VocabularyId, Vocabulary> = {
  flowers: {
    id: "flowers",
    description:
      "Species of flower or flowering plant — 102 garden and wild species, e.g. corn poppy, oxeye daisy, sunflower, rose.",
    instanceNoun: "flower",
    labels: FLOWERS_102,
    hypothesis: "a photo of a {}, a type of flower",
  },
  objects: {
    id: "objects",
    description:
      "Everyday objects, animals, vehicles, food and people — 80 common categories, e.g. person, dog, car, pizza, laptop.",
    instanceNoun: "object",
    labels: COCO_80,
    hypothesis: "a photo of a {}",
  },
};

export { COCO_80, FLOWERS_102 };
