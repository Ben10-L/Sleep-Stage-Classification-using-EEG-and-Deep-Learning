PYTHON ?= python
DATA ?= data/processed/sleep_edf_trimmed.npz

.PHONY: test figures demo inspect trim ml cnn lstm transformer report

test:
	$(PYTHON) -m pytest -q
figures:
	$(PYTHON) scripts/generate_lesson_figures.py
demo:
	$(PYTHON) scripts/run_pipeline.py demo
inspect:
	$(PYTHON) scripts/run_pipeline.py inspect --data $(DATA)
trim:
	$(PYTHON) scripts/run_pipeline.py trim
ml:
	$(PYTHON) scripts/run_pipeline.py ml --data $(DATA) --output results/real/ml
cnn:
	$(PYTHON) scripts/run_pipeline.py deep --data $(DATA) --model cnn --epochs 20 --batch-size 16 --device auto --output results/real/cnn
lstm:
	$(PYTHON) scripts/run_pipeline.py deep --data $(DATA) --model lstm --epochs 20 --batch-size 16 --device auto --output results/real/lstm
transformer:
	$(PYTHON) scripts/run_pipeline.py deep --data $(DATA) --model transformer --epochs 20 --batch-size 16 --device auto --output results/real/transformer
report:
	$(PYTHON) scripts/run_pipeline.py report --results results/real --data $(DATA) --output results/real/EXPERIMENT_REPORT.md
