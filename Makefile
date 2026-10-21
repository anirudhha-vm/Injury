.PHONY: paper-pdf paper-results fetch-data real-growth real-risk setup test test-slow lint gate sim sim-study-growth sim-study-sizes evaluate ablation dashboard-data dashboard pipeline methods paper-figs todo

# A system PYTHONPATH (e.g. from a ROS install) leaks foreign pytest plugins into the venv.
unexport PYTHONPATH

setup:  ## install pinned environment + git hooks
	uv sync
	uv run pre-commit install

test:  ## unit/integration tests (excludes human_gate tests)
	uv run pytest

test-slow:  ## tests that run the MCMC sampler (minutes)
	uv run pytest -m slow --no-cov

lint:  ## ruff + mypy --strict + paper/ results guard
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy
	uv run python scripts/check_no_hardcoded_results.py

gate:  ## human-verification gates; FAILS BY DESIGN until each TODO(human) is resolved
	uv run pytest -m human_gate --no-cov

todo:  ## list everything waiting on a human decision
	@grep -rn "TODO(human)" --include="*" --exclude-dir=.git --exclude-dir=.venv . || true

sim:  ## generate the synthetic cohort into data/synthetic/ (flagged synthetic=True)
	uv run python scripts/make_synthetic_cohort.py

sim-study-growth:  ## simulation study for the growth twin (SYNTHETIC; tens of minutes)
	uv run python scripts/sim_study_growth.py

sim-study-sizes:  ## performance of the tabular models against cohort size (SYNTHETIC)
	uv run python scripts/sim_study_sizes.py

paper-pdf:  ## IEEE manuscript: generated values/tables/figures, then PDF (needs Tectonic; TECTONIC=path)
	uv run python scripts/build_paper.py

paper-results:  ## draft paper/results_draft.md from the latest runs (every number tagged with its run)
	uv run python scripts/draft_results.py

fetch-data:  ## download the public datasets (Berkeley growth, runners' injuries) with checksums
	uv run python scripts/fetch_public_data.py

real-growth:  ## growth twin on the real Berkeley growth data (hours)
	uv run python scripts/real_growth_berkeley.py

real-risk:  ## injury-risk evaluation on the real runners' dataset
	uv run python scripts/real_risk_runners.py

evaluate:  ## out-of-fold evaluation of the tabular models (SYNTHETIC)
	uv run python scripts/evaluate.py

ablation:  ## the nine-configuration ablation grid (SYNTHETIC; slow)
	uv run python scripts/ablation.py

dashboard-data:  ## build the dashboard bundle from the synthetic cohort and saved growth posteriors
	uv run python scripts/build_dashboard.py

dashboard:  ## open the local research dashboard in a browser
	uv run streamlit run apps/twin_dashboard/app.py

pipeline:  ## synthetic cohort -> evaluation -> paper figures and tables
	$(MAKE) sim evaluate paper-figs

methods:  ## regenerate paper/methods_draft.md from the configs
	uv run python scripts/draft_methods.py

paper-figs:  ## regenerate every figure and table in paper/ from the latest runs
	uv run python scripts/make_figures.py
