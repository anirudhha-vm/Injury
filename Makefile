.PHONY: setup test lint gate sim pipeline paper-figs todo

# A system PYTHONPATH (e.g. from a ROS install) leaks foreign pytest plugins into the venv.
unexport PYTHONPATH

setup:  ## install pinned environment + git hooks
	uv sync
	uv run pre-commit install

test:  ## unit/integration tests (excludes human_gate tests)
	uv run pytest

lint:  ## ruff + mypy --strict + paper/ results guard
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy
	uv run python scripts/check_no_hardcoded_results.py

gate:  ## human-verification gates; FAILS BY DESIGN until each TODO(human) is resolved
	uv run pytest -m human_gate --no-cov

todo:  ## list everything waiting on a human decision
	@grep -rn "TODO(human)" --include="*" --exclude-dir=.git --exclude-dir=.venv . || true

sim:  ## generate the synthetic cohort (M1)
	@echo "make sim: not implemented until milestone M1"; exit 1

pipeline:  ## end-to-end pipeline on synthetic data (M5-M6)
	@echo "make pipeline: not implemented until milestone M6"; exit 1

paper-figs:  ## regenerate every figure/table in paper/ (M7)
	@echo "make paper-figs: not implemented until milestone M7"; exit 1
