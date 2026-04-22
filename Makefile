SHELL := /usr/bin/env bash

PY_ULTRA ?= .venv-ultra/bin/python
PY_V5 ?= .venv-v5/bin/python

.PHONY: help dataset-check train-v8 train-v11 train-latest train-v5 run-all compare-test slurm-ultra slurm-v5 slurm-ultra-p100 slurm-v5-p100 clean-artifacts

help:
	@echo "Available targets:"
	@echo "  dataset-check     - Validate YOLO dataset integrity"
	@echo "  train-v8          - Train Ultralytics v8"
	@echo "  train-v11         - Train Ultralytics v11"
	@echo "  train-latest      - Train Ultralytics latest"
	@echo "  train-v5          - Train YOLOv5"
	@echo "  run-all           - Smoke run all families with short epochs"
	@echo "  compare-test      - Compare test metrics across families"
	@echo "  slurm-ultra       - Submit standard Ultralytics Slurm job"
	@echo "  slurm-v5          - Submit standard YOLOv5 Slurm job"
	@echo "  slurm-ultra-p100  - Submit P100 Ultralytics Slurm job"
	@echo "  slurm-v5-p100     - Submit P100 YOLOv5 Slurm job"
	@echo "  clean-artifacts   - Remove generated logs and runs"

dataset-check:
	$(PY_ULTRA) tools/check_dataset.py --data configs/data.caries.yaml

train-v8:
	$(PY_ULTRA) -m scripts.train.train_ultralytics --family v8

train-v11:
	$(PY_ULTRA) -m scripts.train.train_ultralytics --family v11

train-latest:
	$(PY_ULTRA) -m scripts.train.train_ultralytics --family latest

train-v5:
	$(PY_V5) -m scripts.train.train_yolov5 --yolov5-dir third_party/yolov5

run-all:
	$(PY_ULTRA) -m scripts.train.run_all --targets v5 v8 v11 latest --epochs 1

compare-test:
	$(PY_ULTRA) -m scripts.eval.compare_test_models --models v5 v8 v11 latest

slurm-ultra:
	sbatch scripts/slurm/train_ultralytics.sbatch

slurm-v5:
	sbatch scripts/slurm/train_yolov5.sbatch

slurm-ultra-p100:
	sbatch scripts/slurm/train_ultralytics_p100.sbatch

slurm-v5-p100:
	sbatch scripts/slurm/train_yolov5_p100.sbatch

clean-artifacts:
	rm -rf runs/* logs/*
	mkdir -p logs/slurm
