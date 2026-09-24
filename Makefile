# The whole system, reproducible from nothing.
#
# Every artifact this project ships is derived: crops from a dataset,
# embeddings from crops, a probe from embeddings, a claim from a probe. Written
# as shell scripts that chain is a thing only its author can run; written as
# rules it is a thing anyone can re-derive, and make skips the stages whose
# inputs have not changed.
#
#   make check     everything a pull request must pass
#   make flowers   re-derive the flower probe from the dataset
#   make coco      re-derive the object probe from COCO val2017
#   make clean     remove derived artifacts, keep downloads

WORK ?= .work
NODE := node --experimental-strip-types
PY ?= python3
SPLITS := train calib test

.PHONY: check test lint types build eval eval-offline flowers coco clean distclean help

help:
	@grep -E '^#   make' $(MAKEFILE_LIST) | sed 's/^#   /  /'

# ---------------------------------------------------------------- verification

check: types lint test eval-offline
	@echo "\nEverything a pull request must pass is green."

types:
	npx tsc --noEmit

lint:
	npx eslint src evals

test:
	npm test

build:
	npm run build

eval:
	npm run eval

eval-offline:
	npm run eval:offline

# ----------------------------------------------------------------- flower probe
# Oxford Flowers-102 via HuggingFace: the Oxford mirror serves at ~300 KB/s
# against ~7 MB/s, which is a five-hour wait against a five-minute one.

FLOWERS := $(WORK)/flowers
FLOWERS_SHARDS := $(FLOWERS)/shards/.fetched
FLOWERS_BINS := $(foreach s,$(SPLITS),$(FLOWERS)/$(s).bin)

flowers: public/probes/flowers.json

$(FLOWERS_SHARDS):
	@mkdir -p $(FLOWERS)/shards
	@echo "fetching Flowers-102 …"
	@base=https://huggingface.co/datasets/dpdl-benchmark/oxford_flowers102/resolve/main/data; \
	for f in train-00000-of-00001 validation-00000-of-00001 \
	         test-00000-of-00006 test-00001-of-00006 test-00002-of-00006 \
	         test-00003-of-00006 test-00004-of-00006 test-00005-of-00006; do \
	  [ -f $(FLOWERS)/shards/$$f.parquet ] || curl -sL "$$base/$$f.parquet" -o $(FLOWERS)/shards/$$f.parquet; \
	done
	@touch $@

$(FLOWERS)/train.json: $(FLOWERS_SHARDS)
	$(PY) tools/from_parquet.py $(FLOWERS)/shards $(FLOWERS) --all

$(FLOWERS)/%.bin: $(FLOWERS)/train.json
	$(NODE) tools/embed.mjs $(FLOWERS)/$*.json $@

$(FLOWERS)/text.bin: $(FLOWERS)/train.json
	$(NODE) tools/embed-text.mjs flowers $@

public/probes/flowers.json: $(FLOWERS_BINS) $(FLOWERS)/text.bin
	$(PY) tools/fit.py $(FLOWERS) public/probes
	@cp $(FLOWERS)/report.json docs/probe-report.json

flowers-curve: $(FLOWERS_BINS)
	$(PY) tools/curve.py $(FLOWERS)

# ----------------------------------------------------------------- object probe
# COCO val2017 only: 5k images and ~36k labelled instances is enough to fit a
# linear probe, and the train set is 20 GB for no gain at this model size.

COCO := $(WORK)/coco
COCO_DATA := $(COCO)/done.flag
COCO_BINS := $(foreach s,$(SPLITS),$(COCO)/$(s).bin)

coco: public/probes/objects.json

$(COCO_DATA):
	@mkdir -p $(COCO)
	@echo "fetching COCO val2017 …"
	cd $(COCO) && curl -sL -o annotations.zip http://images.cocodataset.org/annotations/annotations_trainval2017.zip \
	  && unzip -qo annotations.zip annotations/instances_val2017.json && rm -f annotations.zip \
	  && curl -sL -o val2017.zip http://images.cocodataset.org/zips/val2017.zip \
	  && unzip -qo val2017.zip && rm -f val2017.zip
	@touch $@

$(COCO)/train.json: $(COCO_DATA)
	$(PY) tools/coco_crops.py $(COCO) $(COCO)

$(COCO)/%.bin: $(COCO)/train.json
	$(NODE) tools/embed.mjs $(COCO)/$*.json $@

$(COCO)/text.bin: $(COCO)/train.json
	$(NODE) tools/embed-text.mjs objects $@

public/probes/objects.json: $(COCO_BINS) $(COCO)/text.bin
	$(PY) tools/fit.py $(COCO) public/probes --name objects

coco-curve: $(COCO_BINS)
	$(PY) tools/curve.py $(COCO)

# ---------------------------------------------------------------------- tidying

clean:
	rm -rf $(WORK)/*/[tc]*.bin $(WORK)/*/*.json .next

distclean: clean
	rm -rf $(WORK)
