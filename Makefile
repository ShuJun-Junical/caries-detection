SHELL := /usr/bin/env bash

PY_ULTRA ?= .venv-ultra/bin/python
PY_V5 ?= .venv-v5/bin/python
DEVICE ?= 0
WORKERS ?= 8

.PHONY: help dataset-check train-v8 train-v11 train-v26 train-v5 run-all compare-test slurm-ultra slurm-v5 slurm-ultra-p100 slurm-v5-p100 clean-artifacts

help:
	@echo "Available targets:"
	@echo "  dataset-check     - Validate YOLO dataset integrity"
	@echo "  train-v8          - Train Ultralytics v8"
	@echo "  train-v11         - Train Ultralytics v11"
	@echo "  train-v26         - Train Ultralytics YOLO26"
	@echo "  train-v5          - Train YOLOv5"
	@echo "  run-all           - Sequentially run all training families"
	@echo "  compare-test      - Compare test metrics across families"
	@echo "  slurm-ultra       - Submit standard Ultralytics Slurm job"
	@echo "  slurm-v5          - Submit standard YOLOv5 Slurm job"
	@echo "  slurm-ultra-p100  - Submit P100 Ultralytics Slurm job"
	@echo "  slurm-v5-p100     - Submit P100 YOLOv5 Slurm job"
	@echo "  clean-artifacts   - Remove generated logs and runs"

dataset-check:
	$(PY_ULTRA) tools/check_dataset.py --data configs/data.caries.yaml

train-v8:
	$(PY_ULTRA) -m scripts.train.train_ultralytics --family v8 --device $(DEVICE) --workers $(WORKERS)

train-v11:
	$(PY_ULTRA) -m scripts.train.train_ultralytics --family v11 --device $(DEVICE) --workers $(WORKERS)

train-v26:
	$(PY_ULTRA) -m scripts.train.train_ultralytics --family v26 --device $(DEVICE) --workers $(WORKERS)

train-v5:
	$(PY_V5) -m scripts.train.train_yolov5 --yolov5-dir third_party/yolov5 --device $(DEVICE) --workers $(WORKERS)

run-all:
	$(PY_ULTRA) -m scripts.train.run_all --targets v5 v8 v11 v26 --device $(DEVICE) --workers $(WORKERS)

compare-test:
	$(PY_ULTRA) -m scripts.eval.compare_test_models --models v5 v8 v11 v26

slurm-ultra:
	sbatch scripts/slurm/train_ultralytics.sbatch

slurm-v5:
	sbatch scripts/slurm/train_yolov5.sbatch

slurm-ultra-p100:
	sbatch scripts/slurm/train_ultralytics_p100.sbatch

slurm-v5-p100:
	sbatch scripts/slurm/train_yolov5_p100.sbatch

clean-artifacts:
	@targets="$$(find runs logs -mindepth 1 -print 2>/dev/null | sort)"; \
	if [[ -z "$$targets" ]]; then \
		echo "No artifacts to remove under runs/ or logs/."; \
		mkdir -p runs logs/slurm; \
		exit 0; \
	fi; \
	echo "The following artifacts will be removed:"; \
	printf '%s\n' "$$targets"; \
	read -r -p "Delete these artifacts? [y/N] " confirm; \
	if [[ "$$confirm" != "y" && "$$confirm" != "Y" ]]; then \
		echo "Aborted."; \
		exit 0; \
	fi; \
	rm -rf runs/* logs/*; \
	mkdir -p runs logs/slurm; \
	echo "Post-cleanup artifact directories:"; \
	find runs logs -maxdepth 2 -mindepth 1 -print | sort
