# Common tasks. Run from the repository root.
PY ?= python3

check:            ## unit checks, no models or API key needed
	$(PY) -m pytest

smoke:            ## pipeline with the LLM stubbed; needs YOLO_MODEL_PATH
	$(PY) -m research.smoke_pipeline

audit:            ## dataset statistics; set DS to a Roboflow YOLO export
	$(PY) -m research.dataset_audit --dataset $(DS) --out out/audit

chapter:          ## run the pipeline on CH (a folder of images); OUT defaults to data/output
	$(PY) scripts/process_chapter.py --input $(CH) --output $(or $(OUT),data/output) --dual-ocr --correction --segmentation --save-segments --translator claude

help:
	@grep -E '^[a-z]+:.*##' Makefile | sed 's/:.*##/ -/'

.PHONY: check smoke audit chapter help
